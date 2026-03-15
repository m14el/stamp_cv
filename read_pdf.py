import pypdf
import sys

def main():
    try:
        with open("Статья.pdf", "rb") as f:
            reader = pypdf.PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            print(text)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
