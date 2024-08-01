
import PyPDF2 

def getPdf(given_pdf):

    # Open the PDF file
    with open(given_pdf, 'rb') as file:
        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfReader(file)
        
        # Initialize a variable to store extracted text
        extracted_text = ""
        
        # Loop through each page and extract text
        for page_num in range(len(pdf_reader.pages)):
            # Get a page object
            page = pdf_reader.pages[page_num]
            
            # Extract text from the page
            text = page.extract_text()
            
            # Append the extracted text to the variable
            extracted_text += text

    with open('catalog_textfile', 'w') as file:
        for item in extracted_text:
            file.write(f"{item}")

    return 'catalog_textfile'

