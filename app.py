from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv()

# Load the model
model = ChatGoogleGenerativeAI(model="gemini-2.5-pro-exp-03-25")

# Define messages
m = [
    SystemMessage(content="You are a helpful dictionary. Tell me the meaning of the inputted word in simple English in a very short and precise form.I mostly work in area related to economics,statestics,machine learning,probability,computer science,investing , trading ,stock market etc")
]

# Set Streamlit page config (MUST be the first command)
st.set_page_config(page_title="AI Study Assistant", layout="wide")
# Streamlit UI setup
st.header("📚 Dictionary 🔎")

# Get user input
user_input = st.text_input("Enter the word: ")
m.append(HumanMessage(content=user_input))


#  Corrected Output format using Pydantic
class Output_formate(BaseModel):
    """Structured output format for word dictionary."""
    
    correct_spelling: str = Field(description="Write down the correct spelling of input. If there is any spelling mistake, just write the correct spelling.")
    meaning: str = Field(description="Short and precise meaning of the word.")
    ex_eng_1: str = Field(description="Example demonstrating use of inputted word in English.")
    ex_eng_2: str = Field(description="Another example demonstrating use of inputted word in English.")
    meaning_hindi: str = Field(description="Short and precise meaning of the word in Hinglish.")
    ex_hindi_1: str = Field(description="Example demonstrating use of inputted word in Hinglish.")
    commonality_score: int = Field(description="How common is the inputted word in spoken English on a scale of 0 to 100.")


# Check if user input is provided before invoking the model
if user_input:
    m.append(HumanMessage(content=user_input))

    # Use structured model
    structured_model = model.with_structured_output(Output_formate)

    # Invoke model with structured output
    result = structured_model.invoke(m)
    
    # Highlight structured output using HTML & CSS
    st.markdown(f"""
    <div style="
          /* Dark Gray Background */
        padding: 15px; 
        border-radius: 12px; 
        box-shadow: 3px 3px 15px rgba(0,0,0,0.5);  /* Deeper Shadow for Depth */
        margin-bottom: 12px;
        ">
        <p style="font-size: 18px; font-weight: bold; margin-bottom: 5px;">
            <span style="color: #99a700;">[{result.commonality_score}]</span> 
            <span style="color: #76a893;">{result.correct_spelling}</span> - 
            <span style="font-style: italic; color: #a0eea0;">{result.meaning}</span>
        </p>
    </div>
    """, unsafe_allow_html=True)



    st.write(f'📚:{result.ex_eng_1}  📚:{result.ex_eng_2}')
    st.write(f'🔡:{result.meaning_hindi} 📚:{result.ex_hindi_1}')
else:
    st.write("Please enter a word to get the meaning.")


