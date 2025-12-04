import pytest
import os
import pandas as pd
import datetime
import uuid
from streamlit.testing.v1 import AppTest
from tests.data_loader import load_questions

# Configuration
# Ensure this path points to your actual Excel file
QUESTIONS_FILE = "tests/input/test_preguntas.xlsx" 
QUESTIONS_COLUMN = "Question"

def test_app_startup():
    """
    Smoke test: Verifies that the app starts successfully without exceptions.
    """
    at = AppTest.from_file("main.py").run()
    assert not at.exception

def test_single_interaction():
    """
    Verifies a single hardcoded interaction works.
    """
    at = AppTest.from_file("main.py").run()
    
    # Simulate user input
    at.chat_input[0].set_value("Hola, -+c+¦mo est+ís?").run()
    
    # Check for response
    # Assuming app.py uses st.session_state.messages to store history
    if "messages" in at.session_state:
        messages = at.session_state.messages
        assert len(messages) >= 2
        assert messages[-1]["role"] == "assistant"
        assert len(messages[-1]["content"]) > 0
    
    assert not at.exception

def test_batch_questions_from_excel():
    """
    Reads questions from an Excel file, runs them against the app, and saves responses to an Excel file.
    """
    questions = load_questions(QUESTIONS_FILE, QUESTIONS_COLUMN)
    
    if not questions:
        pytest.skip(f"No questions found in {QUESTIONS_FILE} or file missing. Skipping batch test.")
    
    print(f"\nLoaded {len(questions)} questions from {QUESTIONS_FILE}")
    
    results = []
    
    for i, question in enumerate(questions):
        print(f"Testing question {i+1}: {question}")
        
        response_text = ""
        status = "Success"
        error_msg = ""
        session_id = str(uuid.uuid4())
        
        try:
            # Start a fresh app instance for each question
            at = AppTest.from_file("main.py").run()
            
            # Input the question
            at.chat_input[0].set_value(question).run(timeout=60) # Increased timeout
            
            if at.exception:
                status = "App Error"
                error_msg = str(at.exception)
            else:
                if "messages" in at.session_state:
                    messages = at.session_state.messages
                    if messages and messages[-1]["role"] == "assistant":
                        response_text = messages[-1]["content"]
                    else:
                        status = "No Response"
                        error_msg = "Last message was not from assistant"
                else:
                    status = "No Session State"
                    error_msg = "Session state 'messages' not found"
                    
        except Exception as e:
            status = "Test Exception"
            error_msg = str(e)
            
        results.append({
            "SessionID": session_id,
            "Question": question,
            "Answer": response_text,
            "Status": status,
            "Error": error_msg,
            "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    # Save results to Excel
    output_dir = "tests/output"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"test_results_{timestamp}.xlsx")
    
    try:
        df = pd.DataFrame(results)
        df.to_excel(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
    except Exception as e:
        print(f"Failed to save results to Excel: {e}")

    # Fail the test if any errors occurred, so CI/CD knows something went wrong
    failures = [r for r in results if r["Status"] != "Success"]
    if failures:
        pytest.fail(f"{len(failures)} questions failed. See {output_file} for details.")
