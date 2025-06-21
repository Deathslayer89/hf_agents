import os
import gradio as gr
import requests
import pandas as pd
from agent_smolagents import SmolagentsAgent
import random
from dotenv import load_dotenv


load_dotenv()
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

def get_agent_info():
    """Get agent code information at startup."""
    space_id = os.getenv("SPACE_ID")
    if space_id:
        agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
        return f"**Agent Code Repository:** {agent_code}\n**Space ID:** {space_id}"
    else:
        return "**Agent Code Repository:** https://github.com/your-repo\n**Space ID:** Not configured"

def update_profile_info(profile: gr.OAuthProfile | None):
    """Update profile information display when login state changes."""
    if profile:
        return f"**Logged in as:** {profile.username}\n**Name:** {profile.name or 'Not provided'}\n**Profile URL:** https://huggingface.co/{profile.username}"
    else:
        return "**Status:** Not logged in. Please log in to run evaluation."

def run_and_submit_all(profile: gr.OAuthProfile | None):
    """Fetches questions, runs the SmolagentsAgent, and submits answers."""
    if not profile:
        return "Please Login to Hugging Face with the button.", None

    username = profile.username
    space_id = os.getenv("SPACE_ID")
    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    try:
        agent = SmolagentsAgent()
    except Exception as e:
        return f"Error initializing agent: {e}", None
    
    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main" if space_id else "https://github.com/your-repo"

    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        random.shuffle(questions_data)
        
        if not questions_data:
             return "No questions available.", None
    except Exception as e:
        return f"Error fetching questions: {e}", None

    results_log = []
    answers_payload = []
    
    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")
        if not task_id or question_text is None:
            continue
        try:
            submitted_answer = agent(question_text, task_id)
            answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": submitted_answer})
        except Exception as e:
             results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": f"ERROR: {e}"})

    if not answers_payload:
        return "No answers generated.", pd.DataFrame(results_log)

    submission_data = {"username": username, "agent_code": agent_code, "answers": answers_payload}
    
    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        return final_status, pd.DataFrame(results_log)
        
    except Exception as e:
        return f"Submission Failed: {str(e)}", pd.DataFrame(results_log)

with gr.Blocks() as demo:
    gr.Markdown("# GAIA Benchmark Agent Evaluation")
    gr.Markdown(
        """
        **Instructions:**
        1. Log in to your Hugging Face account using the button below
        2. Click 'Run Evaluation & Submit' to fetch questions, run your enhanced multimodal agent, and submit answers
        3. Results will show your score and individual question answers
        
        **Note:** This process may take several minutes as the agent processes all questions.
        """
    )
    
    # Agent Information Section
    gr.Markdown("## Agent Information")
    agent_info_display = gr.Markdown(get_agent_info())
    
    # Profile Information Section
    gr.Markdown("## Profile Information")
    profile_info_display = gr.Markdown(update_profile_info(None))
    
    # Login and Evaluation Section
    gr.Markdown("## Evaluation")
    login_button = gr.LoginButton()
    run_button = gr.Button("Run Evaluation & Submit All Answers")
    status_output = gr.Textbox(label="Status / Results", lines=5, interactive=False)
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)

    # Update profile info when login state changes
    login_button.click(
        fn=update_profile_info,
        outputs=[profile_info_display]
    )
    
    run_button.click(
        fn=run_and_submit_all,
        outputs=[status_output, results_table]
    )

if __name__ == "__main__":
    print(get_agent_info())
    demo.launch(debug=True, share=False)