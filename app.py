import os
import pandas as pd
import pickle
from pypdf import PdfReader
import re
import streamlit as st

# Load models
word_vector = pickle.load(open("tfidf.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)  
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText) 
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText

category_mapping = {
    8: "Python Developer",
       17: " Java Developer",                        
       0: "Testing",                               
       1: "Backend Developer",                     
       31: "Cloud Engineer",                        
       22: "DevOps Engineer",                      
       29: "Frontend Developer",                    
       19: "Data Scientist",                        
       6: "Full Stack Developer",                  
       20: " Mobile App Developer (iOS/Android)",    
       5: "Web Designing",                         
       28: "HR",                                    
       2: "Machine Learning Engineer",             
       14: "Hadoop",                                
       25: "Sales",                                 
       27: "Operations Manager",                   
       11: "Data Science",                          
       24: "Mechanical Engineer",                   
       26: "Blockchain",                            
       10: "ETL Developer",                         
       18: "Arts",                                  
       13: "Database",                              
       12: "Health and fitness",                    
       4: "Electrical Engineering",                
       30: "PMO",                                   
       15: "Business Analyst",                      
       3: "DotNet Developer",                      
       9: "Automation Testing",                    
       23: "Network Security Engineer,",             
       21: "SAP Developer,",
       7: "Civil Engineer "
}

def categorize_resumes(uploaded_files, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    results = []
    
    for uploaded_file in uploaded_files:
        if uploaded_file.name.endswith('.pdf'):  # Change the extension as needed
            reader = PdfReader(uploaded_file)
            page = reader.pages[0]
            text = page.extract_text()
            cleaned_resume = cleanResume(text)

            input_features = word_vector.transform([cleaned_resume])
            prediction_id = model.predict(input_features)[0]
            category_name = category_mapping.get(prediction_id, "Unknown")
            
            category_folder = os.path.join(output_directory, category_name)
            
            if not os.path.exists(category_folder):
                os.makedirs(category_folder)
            
            target_path = os.path.join(category_folder, uploaded_file.name)
            with open(target_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            results.append({'filename': uploaded_file.name, 'category': category_name})
    
    results_df = pd.DataFrame(results)
    return results_df

st.title("Resume Categorizer Application")
st.subheader("With Python & Machine Learning")

uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
output_directory = st.text_input("Output Directory", "categorized_resumes")

if st.button("Categorize Resumes"):
    if uploaded_files and output_directory:
        results_df = categorize_resumes(uploaded_files, output_directory)
        st.write(results_df)
        results_csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download results as CSV",
            data=results_csv,
            file_name='categorized_resumes.csv',
            mime='text/csv',
        )
        st.success("Resumes categorization and processing completed.")
    else:
        st.error("Please upload files and specify the output directory.")