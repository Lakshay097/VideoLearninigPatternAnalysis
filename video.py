import os, re, json, yaml, shutil, textwrap, tempfile, subprocess
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from gtts import gTTS
from pydub import AudioSegment

load_dotenv()

def sanitize_manim_code(raw_code: str) -> str:
    code = (raw_code or "").strip()
    
    # Replace external assets + fix LLM errors
    code = re.sub(r"SVGMobject\([^)]*\)", "Circle()", code)
    code = re.sub(r"ImageMobject\([^)]*\)", "Rectangle()", code)
    code = re.sub(r"GrowArrow\([^)]*\)", "Create(arrow)", code)
    code = re.sub(r"scale_tips\s*=\s*True", "", code)
    code = re.sub(r'Text\("([^"]*)"\)', r'Text("\1", font_size=32)', code)
    
    # Auto-layout
    lines = code.splitlines()
    has_title = any(re.match(r"\s*title\s*=\s*Text\(", ln) for ln in lines)
    out = []
    
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("#"):
            out.append(s)
            continue
        
        if re.match(r"title\s*=\s*Text\(", s):
            if not any(tok in s for tok in (".to_edge(", ".shift(", ".move_to(", ".align_to(", ".next_to(")):
                s += ".to_edge(UP)"
            out.append(s)
            continue
        
        if re.search(r"=\s*Text\(", s) and not re.match(r"title\s*=", s):
            if not any(tok in s for tok in (".next_to(", ".to_edge(", ".shift(", ".move_to(", ".align_to(")):
                s += ".next_to(title, DOWN)" if has_title else ".to_edge(UP)"
            out.append(s)
            continue
        
        if re.search(r"=\s*(Circle|Square|Rectangle)\(", s):
            if not any(tok in s for tok in (".shift(", ".move_to(", ".next_to(", ".to_edge(", ".arrange(")):
                s += ".shift(DOWN*0.5)"
            out.append(s)
            continue
        
        out.append(s)
    
    return "\n".join(out)

def load_style_profile(path: str | None) -> dict:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Style profile not found: {p}")
    raw = p.read_text(encoding="utf-8").strip()
    try:
        obj = yaml.safe_load(raw)
        return obj if isinstance(obj, dict) else {"notes": raw}
    except:
        return {"notes": raw}

def _strip_llm_fences(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"^```(?:yaml|yml|json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```$", "", t)
    if t:
        first_line = t.splitlines()[0]
        t = re.sub(r"^\s*(yaml|yml)\s*$", "", first_line, flags=re.IGNORECASE) + "\n" + "\n".join(t.splitlines()[1:])
    return t.strip()

def parse_blueprint(blueprint_yaml: str) -> dict:
    cleaned = _strip_llm_fences(blueprint_yaml)
    try:
        parsed = yaml.safe_load(cleaned)
        if isinstance(parsed, dict) and "scenes" in parsed:
            return parsed
    except:
        pass
    
    print("YAML parse failed, using regex fallback...")
    scenes = []
    lines = cleaned.splitlines()
    current_scene, current_code = None, []
    
    for line in lines:
        ls = line.strip()
        if re.search(r"\bname:\s*Scene\d+\b", ls):
            if current_scene:
                scenes.append({"name": current_scene, "code": "\n".join(current_code).strip()})
            m = re.search(r"Scene\d+", ls)
            current_scene = m.group(0) if m else None
            current_code = []
            continue
        
        if current_scene and ls and "code:" not in ls:
            if any(tok in ls for tok in ("=", "self.play", "Text(", "Circle(", "Arrow(", "CurvedArrow(", "FadeIn", "Create(", "Write(", "Transform(")):
                current_code.append(ls)
    
    if current_scene:
        scenes.append({"name": current_scene, "code": "\n".join(current_code).strip()})
    
    return {"scenes": scenes}

def setup_chains():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY missing from .env")
    
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1, api_key=api_key)
    
    parser = JsonOutputParser()
    script_prompt = PromptTemplate(
        input_variables=["topic", "style", "style_profile"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
        template='Create a 3-scene explainer for "{topic}".\nRequested style: {style}\nManual style profile: {style_profile}\n\nRules:\n- Exactly 3 scenes.\n- Each scene narration 45-60 words.\n- Visual is short diagram description.\n- Return VALID JSON only.\n\n{format_instructions}\n\n{{"scenes":[{{"narration":"...","visual":"..."}},{{"narration":"...","visual":"..."}},{{"narration":"...","visual":"..."}}]}}'
    )
    script_chain = script_prompt | llm | parser
    
    blueprint_prompt = PromptTemplate(
        input_variables=["script", "style_profile"],
        template="Script JSON: {script}\nStyle profile: {style_profile}\n\nReturn RAW VALID YAML ONLY (2-space indent). NO markdown.\nUse PRIMITIVES ONLY: Text, Circle, Square, Rectangle, Line, Arrow, CurvedArrow.\nSchema:\nscenes:\n  - name: Scene1\n    code: |\n      title = Text(\"Scene Title\", font_size=28).to_edge(UP)\n      left = Circle().shift(UP*0.5 + LEFT*2).set_fill(BLUE, opacity=0.7)\n      right = Circle().shift(UP*0.5 + RIGHT*2).set_fill(GREEN, opacity=0.7)\n      arrow = CurvedArrow(left.get_right(), right.get_left())\n      self.play(Write(title), FadeIn(left), FadeIn(right), Create(arrow))\n\nCRITICAL: title.to_edge(UP), font_size 24-32, labels.next_to() or .shift(), NO SVGMobject."
    )
    blueprint_chain = blueprint_prompt | llm | StrOutputParser()
    
    return script_chain, blueprint_chain

def _find_manim_output_mp4(temp_dir: Path, scene_class_name: str) -> Path | None:
    target = f"{scene_class_name}.mp4"
    for root, _, files in os.walk(temp_dir):
        if "media" in root and target in files:
            return Path(root) / target
    return None

def _prebuild_tts(script: dict, temp_dir: Path):
    audio_paths, durations = [], []
    for i, scene in enumerate(script.get("scenes", [])):
        narration = scene.get("narration", "").strip()
        audio = temp_dir / f"audio{i+1}.mp3"
        
        if not narration:
            durations.append(12.0)
            audio_paths.append(audio)
            continue
        
        gTTS(narration, lang="en", slow=False).save(str(audio))
        
        try:
            seg = AudioSegment.from_mp3(str(audio)).normalize()
            audio_norm = temp_dir / f"audio{i+1}_norm.mp3"
            seg.export(str(audio_norm), format="mp3")
            use_audio = audio_norm
            dur = max(6.0, len(seg) / 1000.0)
            audio.unlink(missing_ok=True)
        except:
            print(f"Audio normalize failed scene {i+1}, using raw")
            use_audio = audio
            try:
                dur = max(6.0, len(AudioSegment.from_mp3(str(audio))) / 1000.0)
            except:
                dur = 12.0
        
        audio_paths.append(use_audio)
        durations.append(float(dur))
    return audio_paths, durations

def _render_scenes(blueprint: dict, temp_dir: Path, durations: list[float]):
    good = 0
    for i, scene in enumerate(blueprint.get("scenes", [])):
        wait_s = durations[i] if i < len(durations) else 12.0
        for retry in range(3):
            code = textwrap.indent(sanitize_manim_code(scene.get("code", "")), "        ")
            hold = f"        self.wait({wait_s:.1f})\n        self.play(*[FadeOut(m) for m in self.mobjects])\n        self.wait(0.2)"
            
            manim_code = f"""from manim import *
config.pixel_height = 720
config.pixel_width = 1280
config.frame_rate = 30
class Scene{i+1}(Scene):
    def construct(self):
{code}
{hold}"""
            
            py_file = temp_dir / f"temp_s{i+1}.py"
            expected_mp4 = temp_dir / f"scene{i+1}.mp4"
            py_file.write_text(manim_code)
            
            cmd = ["manim", "-ql", "--resolution=1280,720", "--disable_caching", str(py_file), f"Scene{i+1}", "--format=mp4"]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=temp_dir)
                if result.returncode == 0:
                    produced = _find_manim_output_mp4(temp_dir, f"Scene{i+1}")
                    if produced.exists():
                        shutil.move(str(produced), str(expected_mp4))
                        print(f"Scene{i+1} SAVED")
                        good += 1
                        break
            except:
                pass
            
            print(f"Scene{i+1} retry {retry+1} failed")
    
    print(f"Render: {good}/{len(blueprint.get('scenes', []))} saved")

def _assemble_video_with_audio(temp_dir: Path, output_dir: str, audio_paths: list[Path]):
    clips_dir = Path(temp_dir)
    dubbed = []
    
    for i, audio in enumerate(audio_paths):
        video_in = clips_dir / f"scene{i+1}.mp4"
        if not (video_in.exists() and audio.exists()):
            continue
        
        dubbed_out = clips_dir / f"dubbed{i+1}.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-i", str(video_in), "-i", str(audio),
            "-c:v", "libx264", "-c:a", "aac", "-shortest", "-pix_fmt", "yuv420p", str(dubbed_out)
        ], check=False, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        
        if dubbed_out.exists():
            dubbed.append(dubbed_out)
    
    if dubbed:
        final_mp4 = Path(output_dir) / "final.mp4"
        concat_txt = clips_dir / "concat.txt"
        with open(concat_txt, "w") as f:
            for d in dubbed:
                f.write(f"file '{d.absolute().as_posix()}'\n")
        
        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_txt),
            "-c:v", "libx264", "-preset", "fast", "-c:a", "aac", "-pix_fmt", "yuv420p", str(final_mp4)
        ], capture_output=True, text=True)
        
        concat_txt.unlink(missing_ok=True)
        print(f"Final video: {final_mp4}")

def generate_video(topic: str, style: str = "simple visual", output_dir: str = "clips", style_profile: dict = None, keep_scenes: bool = False):
    print(f"Generating '{topic}'...")
    
    script_chain, blueprint_chain = setup_chains()
    
    script = script_chain.invoke({"topic": topic, "style": style, "style_profile": json.dumps(style_profile or {}, ensure_ascii=False)})
    blueprint_yaml = blueprint_chain.invoke({"script": json.dumps(script), "style_profile": json.dumps(style_profile or {})})
    blueprint = parse_blueprint(blueprint_yaml)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        audio_paths, durations = _prebuild_tts(script, temp_path)
        _render_scenes(blueprint, temp_path, durations)
        
        if keep_scenes:
            for i in range(1, 4):
                p = temp_path / f"scene{i}.mp4"
                if p.exists():
                    shutil.copy2(p, Path(output_dir) / f"scene{i}.mp4")
        
        _assemble_video_with_audio(temp_path, output_dir, audio_paths)
    
    final_mp4 = Path(output_dir) / "final.mp4"
    size = final_mp4.stat().st_size / 1e6 if final_mp4.exists() else 0
    print(f"COMPLETE: {final_mp4} ({size:.1f}MB)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AI Video Generator")
    parser.add_argument("--topic", required=True)
    parser.add_argument("--style", default="simple visual")
    parser.add_argument("--style_profile")
    parser.add_argument("--keep_scenes", action="store_true")
    parser.add_argument("--output", default="clips")
    args = parser.parse_args()
    
    style_profile = load_style_profile(args.style_profile) if args.style_profile else {}
    generate_video(args.topic, args.style, args.output, style_profile, args.keep_scenes)
