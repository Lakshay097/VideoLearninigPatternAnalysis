import os
import re
import json
import yaml
import shutil
import argparse
import tempfile
import subprocess
from pathlib import Path
import textwrap
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from gtts import gTTS
from pydub import AudioSegment

load_dotenv()

STYLE_DEFAULTS = {
    "background_color": "#1e1e1e",
    "colors": {"primary": "TEAL", "secondary": "ORANGE", "success": "GREEN"},
    "typography": {"title": 32, "label": 20},
}

def sanitize_manim_code(llm_code: str, style: dict | None = None) -> str:
    code = llm_code.replace("\r\n", "\n").replace("\t", "    ").strip()

    # Remove markdown fences
    code = re.sub(r"```(?:python)?", "", code)
    code = code.replace("```", "").strip()

    # Dedent badly formatted LLM output
    code = textwrap.dedent(code)

    lines = []
    for line in code.splitlines():
        if not line.strip():
            lines.append("")
            continue

        if line.lstrip().startswith(("import ", "from ")):
            continue
        
        if re.match(r"\s{12,}\S", line):
            line = line[4:]

        lines.append(line)

    clean = "\n".join(lines).strip()

    forced = "\n".join(
        line if not line.strip() else " " * 8 + line.lstrip()
        for line in clean.splitlines()
    )

    return forced + "\n"

def deep_merge(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base

def load_style(style_arg: str) -> dict:
    if style_arg.endswith((".yml", ".yaml")):
        path = Path(style_arg)
        if not path.exists():
            raise FileNotFoundError(f"Style file not found: {path}")
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    return {}

def setup_chains():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY missing. Set it in your .env file or environment.")
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",  
        temperature=0.1,
        api_key=api_key
    )
    json_parser = JsonOutputParser()
    script_prompt = PromptTemplate(
        input_variables=["topic", "style"],
        partial_variables={"format_instructions": json_parser.get_format_instructions()},
        template=(
            'Generate a 3-scene JSON explainer for the topic "{topic}". '
            'Style: {style}. Exactly 3 scenes, narrations 45-60 words each. JSON only.\n'
            '{format_instructions}'
        )
    )
    script_chain = script_prompt | llm | json_parser

    str_parser = StrOutputParser()
    blueprint_prompt = PromptTemplate(
        input_variables=["script"],
        template=(
            'You are given a JSON script for a 3-scene explainer:\n'
            '{script}\n\n'
            'Generate ONLY this exact YAML structure. 3 scenes with Manim code.\n'
            '```\n'
            'scenes:\n'
            '  - name: Scene 1\n'
            '    code: |\n'
            '      title = Text("Scene Title", font_size=36)\n'
            '      self.play(Write(title))\n'
            '  - name: Scene 2\n'
            '    code: |\n'
            '      circle = Circle(radius=1.5, color=TEAL)\n'
            '      self.play(Create(circle))\n'
            '  - name: Scene 3\n'
            '    code: |\n'
            '      arrow = Arrow(LEFT*1.5, RIGHT*1.5)\n'
            '      self.play(GrowArrow(arrow))\n'
            '```\n'
            'Use Text/Circle/Rectangle/Arrow. Use self.play(Write/Create/GrowArrow). '
            'Do not include explanations or markdown fences.'
        )
    )
    blueprint_chain = blueprint_prompt | llm | str_parser

    return script_chain, blueprint_chain

def parse_blueprint(yaml_str: str) -> dict:

    yaml_str = re.sub(r'^```(?:yaml)?\s*|```$', '', yaml_str.strip(), flags=re.MULTILINE)
    
    idx = yaml_str.find("scenes:")
    if idx >= 0:
        yaml_str = yaml_str[idx:]
    
    try:
        data = yaml.safe_load(yaml_str)
        if not isinstance(data, dict) or 'scenes' not in data:
            raise ValueError("Invalid blueprint structure")
        return data
    except Exception as e:
        print("Blueprint parse failed:", e)
        # For debug, dump raw string
        print("Raw LLM output:", yaml_str[:500], "...")
        return {'scenes': []}

def prebuild_tts(script: dict, temp_dir: Path):
    durations = []
    for i, scene in enumerate(script['scenes']):
        audio_path = temp_dir / f'audio{i+1}.mp3'
        tts = gTTS(scene['narration'])
        tts.save(audio_path)
        seg = AudioSegment.from_mp3(audio_path)
        durations.append(max(3.0, len(seg) / 1000))
    return durations

def render_scene_py(scene_idx: int, code: str, duration: float, style: dict, out_path: Path):
    clean_code = sanitize_manim_code(code, style)
    manim_code = f"""from manim import *
config.pixel_height = 720
config.pixel_width = 1280
config.frame_rate = 30
config.background_color = '{style.get('background_color', '#1e1e1e')}'
class Scene{scene_idx}(Scene):
    def construct(self):
{clean_code}
        self.wait({duration:.1f})
        self.play(*[FadeOut(m) for m in self.mobjects])
"""
    out_path.write_text(manim_code, encoding="utf-8")

def generate_video(topic: str, style: dict, output_dir: str, keep_scenes: bool):
    """Generate 3-scene explainer video with merged style dict."""
    Path(output_dir).mkdir(exist_ok=True)
    raw_dir = Path(output_dir) / 'raw_assets'
    raw_dir.mkdir(parents=True, exist_ok=True)

    video_dir = Path(output_dir) / 'media/videos'
    video_dir.mkdir(parents=True, exist_ok=True)
    
    script_chain, blueprint_chain = setup_chains()
    
    # FIX 3: LLM gets style description (string), not YAML structure
    style_hint = style.get("description", style.get("name", "clean technical explainer"))
    script = script_chain.invoke({
        "topic": topic,
        "style": style_hint  # String for LLM, not dict path
    })
    
    blueprint = None
    for _ in range(3):  # Retries
        blueprint_yaml = blueprint_chain.invoke({'script': json.dumps(script)})
        blueprint = parse_blueprint(blueprint_yaml)
        if blueprint.get('scenes', []):
            break
    
    if not blueprint.get('scenes'):
        raise ValueError('Failed to generate blueprint')
    
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        durations = prebuild_tts(script, tmp_path)
        for i, scene in enumerate(blueprint['scenes']):
            py_path = tmp_path / f'scene{i+1}.py'
            render_scene_py(i+1, scene['code'], durations[i], style, py_path)
            # Render
            subprocess.run(['manim', '-ql', str(py_path), f'Scene{i+1}'], cwd=tmp_path, check=True)
            video_out = video_dir / f'scene{i+1}.mp4'
            scene_media_dir = tmp_path / 'media/videos' / f'Scene{i+1}'
            # Find the mp4 file in the resolution folder
            mp4_files = list(scene_media_dir.glob('*/Scene*.mp4'))
            if not mp4_files:
                raise FileNotFoundError(f"No mp4 file found for scene {i+1} in {scene_media_dir}")
            shutil.move(mp4_files[0], video_out)

            # Dub
            audio_in = tmp_path / f'audio{i+1}.mp3'
            dubbed = video_dir / f'dubbed{i+1}.mp4'
            subprocess.run([
                'ffmpeg', '-y', '-i', str(video_out), '-i', str(audio_in),
                '-c:v', 'copy', '-c:a', 'aac', '-shortest', str(dubbed)
            ], check=True)
        # Concat
        concat_list = '\n'.join(f'file dubbed{i+1}.mp4' for i in range(3))
        list_path = video_dir / 'concat.txt'
        list_path.write_text(concat_list)
        final = Path(output_dir) / 'final.mp4'
        subprocess.run(['ffmpeg', '-f', 'concat', '-safe', '0', '-i', str(list_path), '-c', 'copy', str(final)], check=True)
    
    if keep_scenes:
        shutil.copytree(video_dir, raw_dir / 'scenes')
    print(f'final.mp4 generated: {final} [memory:12]')

def main():
    parser = argparse.ArgumentParser(description="AI Explainer Video Generator")
    parser.add_argument("--topic", required=True)
    parser.add_argument("--style", default="styles/style.yaml")
    parser.add_argument("--output", default="clips")
    parser.add_argument("--keep_scenes", action="store_true")
    args = parser.parse_args()

    style_profile = load_style(args.style)
    style = deep_merge(STYLE_DEFAULTS.copy(), style_profile)

    print(f"🎨 Style loaded: {style.get('name', 'default')}")

    generate_video(
        topic=args.topic,
        style=style,
        output_dir=args.output,
        keep_scenes=args.keep_scenes
    )

if __name__ == '__main__':
    main()
