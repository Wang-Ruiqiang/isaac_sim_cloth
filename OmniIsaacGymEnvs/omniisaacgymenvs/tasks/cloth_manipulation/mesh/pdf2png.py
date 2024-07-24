import fitz  # PyMuPDF

# 打开 PDF 文件
pdf_path = "Checkerboard-A4-25mm-8x6.pdf"
pdf_document = fitz.open(pdf_path)

# 选择要转换的页码，这里选择第一页（页码从0开始）
page_number = 0
page = pdf_document.load_page(page_number)

# 将页面转换为 PNG 图像
image_matrix = fitz.Matrix(2, 2)  # 可以调整矩阵的比例因子以控制输出图像的分辨率
pix = page.get_pixmap(matrix=image_matrix, alpha=False)

# 保存为 PNG 文件
output_image_path = "checkerboard.png"
pix.save(output_image_path)

print(f"Saved image to {output_image_path}")
