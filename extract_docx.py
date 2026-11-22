import docx


def docx_to_txt(docx_path: str, txt_path: str) -> None:
    doc = docx.Document(docx_path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(paragraphs))


# Підстав точні шляхи до своїх .docx-файлів
docx_to_txt("/шлях/до/Misto_upovilnenoi_dii.docx", "data/text1.txt")
docx_to_txt("/шлях/до/dvanadtsiat-obruchiv.docx", "data/text2.txt")