import PyPDF2
import pandas as pd
import nltk
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, dendrogram, cut_tree

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
print("debug")

# Read the text file
def read_text_file(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()
    return data

# Text Preprocessing
def preprocess_text(text):
    # Lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenization
    tokens = nltk.word_tokenize(text, language='english', preserve_line=True)
    
    # Remove stopwords
    stop_words = set(nltk.corpus.stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    # Join tokens back into text
    text = ' '.join(filtered_tokens)
    
    return text

# Feature Extraction using TF-IDF
def extract_features(text_data):
    vectorizer = TfidfVectorizer()
    features = vectorizer.fit_transform(text_data)
    return features, vectorizer

# Hierarchical Clustering
def hierarchical_clustering(features):
    # Calculate the similarity matrix
    similarity_matrix = cosine_similarity(features)
    
    # Apply Agglomerative Hierarchical Clustering
    agg_cluster = linkage(similarity_matrix, method='ward')
    
    return agg_cluster






# Convert to Structured Data
def convert_to_structured_data(text_data, cluster_labels):
    structured_data = pd.DataFrame({
        'Text': text_data,
        'Cluster_Label': cluster_labels
    })
    return structured_data

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

    with open('catalog_textfile.txt', 'w') as file:
        file.write(extracted_text)

    return 'catalog_textfile.txt'

# Main function
def main():

    # Read the text file
    file_path = getPdf('university_course_catalog.pdf')
    print("d")
    text_data = read_text_file(file_path)
    print("debugger")
    
    # Text Preprocessing
    preprocessed_data = [preprocess_text(text) 
        for text in text_data]
    print("debuggerr")

    # Feature Extraction using TF-IDF
    features, vectorizer = extract_features(preprocessed_data)
    print("debuggerrr")

    try:
        agg_cluster = hierarchical_clustering(features)
    except Exception as e:
        print("Error during hierarchical clustering:", e)
        return

    # Cut tree to get clusters
    cluster_labels = cut_tree(agg_cluster, n_clusters=10).reshape(-1,)

   
    

    # Convert to Structured Data
    structured_data = convert_to_structured_data(text_data, cluster_labels)
    structured_data = structured_data.sort_values(by=['Cluster_Label'], ascending=False)
    print(structured_data)

    with open('cluster_text', 'w') as file:
        # Iterate through DataFrame rows
        for index, row in structured_data.iterrows():
            # Convert row to string and write to file
            row_str = '\t'.join(map(str, row)) + '\n'  # Assuming tab-separated values
            file.write(row_str)
    
    structured_data.to_csv('Cluster_data.csv', encoding='utf-8')
    



if __name__ == '__main__':
    main()
