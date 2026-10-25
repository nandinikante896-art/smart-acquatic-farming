"""Usage:  python patch_project.py            (reads app.py, writes app_final.py)
          python patch_project.py other.py"""
import re, sys
src = sys.argv[1] if len(sys.argv) > 1 else "app.py"
c = open(src, encoding="utf-8").read().replace("\r\n", "\n")
if "assistant_bp" in c:
    sys.exit("Already patched. Run the original app.py through this script.")

# 1) load .env (GEMINI_API_KEY) right after Flask config
a = 'app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32MB max upload\n'
assert c.count(a) == 1, "MAX_CONTENT_LENGTH line not found"
c = c.replace(a, a + 'try:\n    from dotenv import load_dotenv\n    load_dotenv(os.path.join(BASE_DIR, ".env"))\nexcept ImportError:\n    pass\n', 1)

# 2) serve the new Home/FISH/PRAWN UI
m = re.search(r"    return render_template_string\(INDEX_HTML,.*\n", c)
assert m, "index() return not found"
c = c[:m.start()] + '    return send_from_directory(os.path.join(BASE_DIR, "static_ui"), "index.html")\n' + c[m.end():]

# 3) /health crash fix (FISH_MODEL_PATH / PRAWN_MODEL_PATH were never defined)
for old, new in [('"fish_weights": FISH_MODEL_PATH,', '"fish_weights": FISH_MODEL_PATHS,'),
                 ('"prawn_weights": PRAWN_MODEL_PATH,', '"prawn_weights": PRAWN_MODEL_PATHS,')]:
    c = c.replace(old, new)

# 4) global error handler turned every 404/405 into a 500 -> return HTTP errors untouched
assert c.count("    raise error\n") == 1, "error handler not found"
c = c.replace("    raise error\n", "    from werkzeug.exceptions import HTTPException\n    if isinstance(error, HTTPException):\n        return error\n    raise error\n", 1)

# 5) register the secure Gemini assistant endpoint
b = 'if __name__ == "__main__":'
assert c.count(b) == 1, "main block not found"
c = c.replace(b, "from assistant_blueprint import bp as assistant_bp\napp.register_blueprint(assistant_bp)\n\n" + b, 1)

open("app_final.py", "w", encoding="utf-8", newline="\n").write(c)
print("OK -> app_final.py created.  Run:  python app_final.py")