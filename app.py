# app.py
import os
import tempfile
from urllib.parse import quote
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from desensitize import desensitize_text

# 尝试导入文档处理库
try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    import openpyxl
    XLSX_AVAILABLE = True
except ImportError:
    XLSX_AVAILABLE = False

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {'txt', 'docx', 'xlsx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/desensitize", methods=["POST"])
def desensitize_api():
    data = request.get_json()
    text = data.get("text", "")
    result = desensitize_text(text)
    return jsonify({"original": text, "desensitized": result})

# ===== 修复版：使用 NamedTemporaryFile + delete=True =====
@app.route("/desensitize-file", methods=["POST"])
def desensitize_file_api():
    if 'file' not in request.files:
        return "❌ 未选择文件", 400
    
    file = request.files['file']
    original_filename = file.filename
    if not original_filename:
        return "❌ 文件名为空", 400

    # 安全提取扩展名
    parts = original_filename.rsplit('.', 1)
    if len(parts) < 2:
        return "❌ 文件没有扩展名，请上传 .txt / .docx / .xlsx 文件", 400
    ext = parts[1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        return "❌ 仅支持 .txt / .docx / .xlsx 文件", 400

    filename = secure_filename(original_filename)

    # 保存原始文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp_in:
        file.save(tmp_in.name)
        input_path = tmp_in.name

    try:
        # 创建临时输出路径（用于写入脱敏后内容）
        output_path = input_path + ".output"

        if ext == 'txt':
            with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            desensitized = desensitize_text(text)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(desensitized)

        elif ext == 'docx':
            if not DOCX_AVAILABLE:
                return "❌ 缺少 python-docx，请安装：pip install python-docx -i https://pypi.tuna.tsinghua.edu.cn/simple", 500
            doc = Document(input_path)
            for para in doc.paragraphs:
                if para.text.strip():
                    para.text = desensitize_text(para.text)
            doc.save(output_path)

        elif ext == 'xlsx':
            if not XLSX_AVAILABLE:
                return "❌ 缺少 openpyxl，请安装：pip install openpyxl -i https://pypi.tuna.tsinghua.edu.cn/simple", 500
            wb = openpyxl.load_workbook(input_path)
            for sheet in wb.sheetnames:
                ws = wb[sheet]
                for row in ws.iter_rows():
                    for cell in row:
                        if cell.value and isinstance(cell.value, str):
                            cell.value = desensitize_text(cell.value)
            wb.save(output_path)

        # 读入内存并返回
        with open(output_path, 'rb') as f:
            file_data = f.read()

        from flask import Response
        download_name = f"脱敏_{filename}"
        response = Response(
            file_data,
            mimetype='application/octet-stream',
            headers={
                "Content-Disposition": f"attachment; filename*=UTF-8''{quote(download_name)}"
            }
        )
        return response

    except Exception as e:
        return f"❌ 处理失败: {str(e)}", 500
    finally:
        # 清理临时文件
        for path in [input_path, output_path]:
            if os.path.exists(path):
                try:
                    os.unlink(path)
                except:
                    pass  # 忽略删除失败（如已被占用）

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)