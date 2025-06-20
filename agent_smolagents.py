import os
import json
import requests
import datetime
from dotenv import load_dotenv
from typing import Dict, List, Any, Optional
from urllib.parse import urlparse
import pandas as pd
import PyPDF2
import openpyxl
from io import BytesIO
from PIL import Image

from smolagents import (
    CodeAgent,
    InferenceClientModel,
    VisitWebpageTool,
    tool
)

load_dotenv()
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

@tool
def multimodal_vision_analysis(image_path: str, question: str = "Describe this image in detail") -> str:
    """Use a multimodal vision-language model to directly analyze images.
    
    Args:
        image_path: Path to the image file or URL
        question: Specific question about the image
        
    Returns:
        Direct analysis from a multimodal VLM
    """
    try:
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
        import torch
        from PIL import Image
        
        # Use a smaller but capable multimodal model for direct image analysis
        model_name = "llava-hf/llava-v1.6-mistral-7b-hf"
        
        try:
            processor = LlavaNextProcessor.from_pretrained(model_name)
            model = LlavaNextForConditionalGeneration.from_pretrained(
                model_name, 
                torch_dtype=torch.float16, 
                low_cpu_mem_usage=True
            )
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            
            if image_path.startswith('http'):
                response = requests.get(image_path, timeout=30)
                image = Image.open(BytesIO(response.content))
            else:
                image = Image.open(image_path)
            
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image"},
                    ],
                },
            ]
            
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                output = model.generate(
                    **inputs, 
                    max_new_tokens=500,
                    do_sample=False,
                    temperature=0.1
                )
            
            response = processor.decode(output[0], skip_special_tokens=True)
            generated_text = response.split("ASSISTANT:")[-1].strip() if "ASSISTANT:" in response else response.strip()
            
            return f"MULTIMODAL VISION ANALYSIS for {image_path}:\n\nQuestion: {question}\n\nResponse: {generated_text}"
            
        except Exception as model_error:
            try:
                model_name_fallback = "microsoft/git-base-coco"
                from transformers import AutoProcessor, AutoModelForCausalLM
                
                processor = AutoProcessor.from_pretrained(model_name_fallback)
                model = AutoModelForCausalLM.from_pretrained(model_name_fallback)
                
                if image_path.startswith('http'):
                    response = requests.get(image_path, timeout=30)
                    image = Image.open(BytesIO(response.content))
                else:
                    image = Image.open(image_path)
                
                inputs = processor(images=image, text=question, return_tensors="pt")
                generated_ids = model.generate(pixel_values=inputs.pixel_values, max_length=150)
                generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                return f"MULTIMODAL VISION ANALYSIS (Fallback) for {image_path}:\n\nQuestion: {question}\n\nResponse: {generated_caption}"
                
            except Exception as fallback_error:
                return f"Error in multimodal vision analysis: Primary model failed ({model_error}), Fallback failed ({fallback_error})"
        
    except ImportError:
        return "Error: Transformers library not available. Install with: pip install transformers torch"
    except Exception as e:
        return f"Error in multimodal vision analysis: {str(e)}"

@tool
def multimodal_vision_qa(image_path: str, question: str) -> str:
    """Ask specific questions about images using multimodal vision models.
    
    Args:
        image_path: Path to the image file or URL
        question: Specific question to ask about the image
        
    Returns:
        Answer to the question from a multimodal vision model
    """
    try:
        result = multimodal_vision_analysis(image_path, question)
        return result.replace("MULTIMODAL VISION ANALYSIS", "MULTIMODAL VISION Q&A")
    except Exception as e:
        return f"Error in multimodal vision Q&A: {str(e)}"

@tool
def tavily_search(query: str, max_results: int = 5) -> str:
    """USE THIS FOR ANYTHING YOU DON'T KNOW AT ALL - when you have no starting point for information. This is your go-to tool for discovering completely unknown topics, current events, recent developments, or when you need to explore what exists about a subject from scratch.
    
    Args:
        query: The search query (supports video URLs, image searches, and general queries)
        max_results: Maximum number of results to return (default: 5)
        
    Returns:
        Formatted search results from Tavily including video transcripts when available
    """
    if not TAVILY_API_KEY:
        return "Error: TAVILY_API_KEY not found in environment variables"
    
    try:
        import requests
        
        url = "https://api.tavily.com/search"
        headers = {
            "Content-Type": "application/json"
        }
        data = {
            "api_key": TAVILY_API_KEY,
            "query": query,
            "max_results": max_results,
            "search_depth": "advanced",
            "include_answer": True,
            "include_raw_content": False,
            "include_images": True
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        
        formatted_results = f"TAVILY SEARCH RESULTS for: {query}\n\n"
        
        if result.get("answer"):
            formatted_results += f"Direct Answer: {result['answer']}\n\n"
        
        if result.get("results"):
            for i, item in enumerate(result["results"], 1):
                title = item.get("title", "No title")
                url_result = item.get("url", "No URL")
                content = item.get("content", "")[:800]
                
                formatted_results += f"Result {i}:\nTitle: {title}\nURL: {url_result}\nContent: {content}\n\n---\n\n"
        
        return formatted_results
    except Exception as e:
        return f"Tavily search failed: {str(e)}"

@tool
def get_youtube_transcript(video_url: str, languages: str = "en") -> str:
    """Extract transcript from a YouTube video URL.
    
    Args:
        video_url: YouTube video URL (e.g., https://www.youtube.com/watch?v=VIDEO_ID)
        languages: Comma-separated language codes (default: "en")
        
    Returns:
        Video transcript text or error message
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        import re
        
        # Extract video ID from URL
        video_id_match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', video_url)
        if not video_id_match:
            return f"Error: Could not extract video ID from URL: {video_url}"
        
        video_id = video_id_match.group(1)
        language_list = [lang.strip() for lang in languages.split(',')]
        
        try:
            # Try to get transcript in requested languages
            transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=language_list)
        except Exception:
            # If requested languages fail, try auto-generated English
            try:
                transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
            except Exception:
                # If that fails, get any available transcript
                transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                transcript_data = transcript_list.find_transcript(['en']).fetch()
        
        # Format transcript
        full_transcript = ""
        for entry in transcript_data:
            text = entry['text'].replace('\n', ' ').strip()
            timestamp = entry['start']
            full_transcript += f"[{timestamp:.1f}s] {text}\n"
        
        if len(full_transcript) > 5000:
            full_transcript = full_transcript[:5000] + "\n... (transcript truncated)"
        
        return f"YOUTUBE TRANSCRIPT for {video_url}:\n\n{full_transcript}"
        
    except ImportError:
        return "Error: youtube-transcript-api package not installed. Run: pip install youtube-transcript-api"
    except Exception as e:
        return f"Error extracting YouTube transcript: {str(e)}"



@tool
def analyze_image(image_path: str, analysis_type: str = "general") -> str:
    """Analyze an image using both multimodal vision models and traditional computer vision.
    
    Args:
        image_path: Path to the image file or URL
        analysis_type: Type of analysis ("general", "text", "objects", "colors", "detailed")
        
    Returns:
        Comprehensive image analysis results
    """
    try:
        # Start with multimodal vision analysis
        if analysis_type == "detailed":
            question = "Provide a detailed analysis of this image, including all objects, people, text, colors, composition, and any other notable elements."
        elif analysis_type == "text":
            question = "Extract and transcribe any text visible in this image."
        elif analysis_type == "objects":
            question = "Identify and list all objects, people, and items visible in this image."
        elif analysis_type == "colors":
            question = "Describe the colors, lighting, and visual style of this image."
        else:
            question = "Describe this image comprehensively, including what you see and any notable details."
        
        vision_result = multimodal_vision_analysis(image_path, question)
        
        # Add traditional computer vision analysis
        try:
            from PIL import Image, ImageStat
            import numpy as np
            
            # Load image
            if image_path.startswith('http'):
                response = requests.get(image_path, timeout=30)
                image = Image.open(BytesIO(response.content))
            else:
                image = Image.open(image_path)
            
            analysis_result = f"\nTRADITIONAL IMAGE ANALYSIS:\n\n"
            analysis_result += f"Image size: {image.size[0]}x{image.size[1]} pixels\n"
            analysis_result += f"Image mode: {image.mode}\n"
            
            if analysis_type in ["general", "colors", "detailed"]:
                # Color analysis
                if image.mode == 'RGB':
                    stat = ImageStat.Stat(image)
                    analysis_result += f"Average RGB: R={stat.mean[0]:.1f}, G={stat.mean[1]:.1f}, B={stat.mean[2]:.1f}\n"
                    
                    # Dominant colors (simplified)
                    image_array = np.array(image)
                    if len(image_array.shape) == 3:
                        avg_color = np.mean(image_array, axis=(0,1))
                        analysis_result += f"Dominant color (RGB): ({avg_color[0]:.0f}, {avg_color[1]:.0f}, {avg_color[2]:.0f})\n"
            
            # Image properties
            if hasattr(image, 'info') and image.info:
                analysis_result += f"Image metadata: {image.info}\n"
            
            # Combine results
            combined_result = f"{vision_result}\n\n---\n\n{analysis_result}"
            return combined_result
            
        except Exception as traditional_error:
            return f"{vision_result}\n\nTraditional analysis failed: {traditional_error}"
        
    except Exception as e:
        return f"Error analyzing image: {str(e)}"

@tool
def arxiv_search(query: str, max_results: int = 3) -> str:
    """Search ArXiv for academic papers.
    
    Args:
        query: Search query for ArXiv papers
        max_results: Maximum number of results to return (default: 3)
        
    Returns:
        Formatted ArXiv search results
    """
    try:
        import urllib.parse
        import urllib.request
        import xml.etree.ElementTree as ET
        
        # Encode the query
        query_encoded = urllib.parse.quote(query)
        
        # ArXiv API URL
        url = f"http://export.arxiv.org/api/query?search_query=all:{query_encoded}&start=0&max_results={max_results}&sortBy=relevance&sortOrder=descending"
        
        # Make the request
        with urllib.request.urlopen(url) as response:
            data = response.read().decode('utf-8')
        
        # Parse XML response
        root = ET.fromstring(data)
        
        # Extract namespace
        namespace = {'atom': 'http://www.w3.org/2005/Atom'}
        
        formatted_results = f"ARXIV SEARCH RESULTS for: {query}\n\n"
        
        entries = root.findall('atom:entry', namespace)
        if not entries:
            return f"No ArXiv papers found for query: {query}"
        
        for i, entry in enumerate(entries, 1):
            title = entry.find('atom:title', namespace)
            title_text = title.text.strip().replace('\n', ' ') if title is not None else "No title"
            
            summary = entry.find('atom:summary', namespace)
            summary_text = summary.text.strip().replace('\n', ' ')[:500] if summary is not None else "No summary"
            
            link = entry.find('atom:id', namespace)
            link_text = link.text if link is not None else "No link"
            
            authors = entry.findall('atom:author', namespace)
            author_names = []
            for author in authors:
                name = author.find('atom:name', namespace)
                if name is not None:
                    author_names.append(name.text)
            authors_text = ", ".join(author_names) if author_names else "No authors"
            
            formatted_results += f"Paper {i}:\nTitle: {title_text}\nAuthors: {authors_text}\nURL: {link_text}\nAbstract: {summary_text}\n\n---\n\n"
        
        return formatted_results
    except Exception as e:
        return f"ArXiv search failed: {str(e)}"

@tool
def wikipedia_search(query: str) -> str:
    """Search Wikipedia for information.
    
    Args:
        query: The search query for Wikipedia
        
    Returns:
        Wikipedia content or search results
    """
    try:
        import requests
        
        # First, search for the article
        search_url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + requests.utils.quote(query)
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; GAIA-Agent/1.0)'
        }
        
        response = requests.get(search_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            title = data.get('title', 'No title')
            extract = data.get('extract', 'No summary available')
            url = data.get('content_urls', {}).get('desktop', {}).get('page', 'No URL')
            
            result = f"WIKIPEDIA RESULT for: {query}\n\n"
            result += f"Title: {title}\n"
            result += f"URL: {url}\n"
            result += f"Summary: {extract}\n"
            
            return result
        else:
            # If direct lookup fails, try search API
            search_api_url = "https://en.wikipedia.org/api/rest_v1/page/search/" + requests.utils.quote(query)
            search_response = requests.get(search_api_url, headers=headers, timeout=10)
            
            if search_response.status_code == 200:
                search_data = search_response.json()
                pages = search_data.get('pages', [])
                
                if pages:
                    # Get the first result
                    first_page = pages[0]
                    page_title = first_page.get('title', query)
                    
                    # Get summary for the first result
                    summary_url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + requests.utils.quote(page_title)
                    summary_response = requests.get(summary_url, headers=headers, timeout=10)
                    
                    if summary_response.status_code == 200:
                        summary_data = summary_response.json()
                        extract = summary_data.get('extract', 'No summary available')
                        url = summary_data.get('content_urls', {}).get('desktop', {}).get('page', 'No URL')
                        
                        result = f"WIKIPEDIA SEARCH RESULT for: {query}\n\n"
                        result += f"Title: {page_title}\n"
                        result += f"URL: {url}\n"
                        result += f"Summary: {extract}\n"
                        
                        return result
            
            return f"No Wikipedia results found for: {query}"
            
    except Exception as e:
        return f"Wikipedia search failed: {str(e)}"



@tool
def download_file(url: str, filename: str = None) -> str:
    """Download a file from a URL and save it locally.
    
    Args:
        url: The URL to download from
        filename: Optional filename to save as
        
    Returns:
        Local path to the downloaded file
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, timeout=30, headers=headers, stream=True)
        response.raise_for_status()
        
        if not filename:
            from urllib.parse import urlparse
            parsed_url = urlparse(url)
            filename = os.path.basename(parsed_url.path) or "downloaded_file"
            if not os.path.splitext(filename)[1]:
                content_type = response.headers.get('content-type', '')
                if 'pdf' in content_type:
                    filename += '.pdf'
                elif 'excel' in content_type or 'spreadsheet' in content_type:
                    filename += '.xlsx'
                elif 'image' in content_type:
                    filename += '.jpg'
                else:
                    filename += '.txt'
        
        os.makedirs("downloads", exist_ok=True)
        filepath = os.path.join("downloads", filename)
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
            
        return f"File downloaded successfully to: {filepath}"
    except Exception as e:
        return f"Failed to download file from {url}: {str(e)}"

@tool
def read_text_file(file_path: str) -> str:
    """Read content from a text file.
    
    Args:
        file_path: Path to the text file
        
    Returns:
        Content of the text file
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            # Return first 3000 characters to avoid overwhelming the context
            return content[:3000] + "..." if len(content) > 3000 else content
    except Exception as e:
        return f"Failed to read text file: {str(e)}"

@tool
def execute_python_code(code: str) -> str:
    """ONLY USE WHEN THE INPUT/TASK IS SURELY KNOWN AND SPECIFIC - for precise calculations, data processing, or when you have exact code to run. Use this tool when you know exactly what computation or processing needs to be done and have specific inputs/data to work with. ALL IMPORTS ARE ALLOWED - no restrictions.
    
    Args:
        code: Python code to execute (any imports allowed, no restrictions)
        
    Returns:
        Execution result or error message
    """
    try:
        import io
        import sys
        import contextlib
        
        # Create string buffer to capture output
        output_buffer = io.StringIO()
        error_buffer = io.StringIO()
        
        # Capture stdout and stderr
        with contextlib.redirect_stdout(output_buffer), contextlib.redirect_stderr(error_buffer):
            # Create a local namespace for execution
            local_namespace = {}
            global_namespace = {
                '__builtins__': __builtins__,
                # Allow all standard library imports and any other imports
            }
            
            # Execute the code - NO IMPORT RESTRICTIONS
            exec(code, global_namespace, local_namespace)
        
        # Get the output
        stdout_content = output_buffer.getvalue()
        stderr_content = error_buffer.getvalue()
        
        result = ""
        if stdout_content:
            result += f"Output:\n{stdout_content}\n"
        if stderr_content:
            result += f"Errors/Warnings:\n{stderr_content}\n"
        
        if not result:
            result = "Code executed successfully (no output)"
            
        return result
        
    except Exception as e:
        return f"Python execution error: {str(e)}"

# ============================================================================
# GAIA QUESTION FETCHING AND RANDOMIZATION (Non-tool functions)
# ============================================================================


# ============================================================================
# ENHANCED SMOLAGENTS AGENT CLASS WITH TRUE MULTIMODAL CAPABILITIES
# ============================================================================

class SmolagentsAgent:
    def __init__(self):
        """Initialize the SmolagentsAgent for GAIA benchmark evaluation."""
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise Exception("HF_TOKEN environment variable is required but not found")
        
        try:
            model_id = "Qwen/Qwen2.5-72B-Instruct"
            self.model = InferenceClientModel(
                model_id=model_id,
                timeout=180,
                token=hf_token
            )
                
        except Exception as e:
            raise Exception(f"Error creating model: {e}")
        
        try:
            enhanced_tools = [
                multimodal_vision_analysis,
                multimodal_vision_qa,
                tavily_search,
                get_youtube_transcript,
                analyze_image,
                arxiv_search,
                wikipedia_search,
                download_file,
                read_text_file,
                execute_python_code
            ]
            
            try:
                enhanced_tools.append(VisitWebpageTool(max_output_length=15000))
            except Exception:
                pass
            
            self.tools = enhanced_tools
            
        except Exception as e:
            raise Exception(f"Error creating tools: {e}")
        
        try:
            self.agent = CodeAgent(
                additional_authorized_imports="*",
                tools=self.tools,
                model=self.model,
                max_steps=15
            )
            
        except Exception as e:
            raise Exception(f"Error creating CodeAgent: {e}")

    def __call__(self, question: str, task_id: str = None) -> str:
        """Process a GAIA question and return an answer."""
        try:
            answer = self.agent.run(question)
            
            if hasattr(answer, 'content'):
                answer = answer.content
            elif isinstance(answer, dict) and 'content' in answer:
                answer = answer['content']
            
            answer = str(answer).strip()
            
            prefixes_to_remove = [
                "FINAL ANSWER:",
                "Final Answer:",
                "Answer:",
                "The answer is:",
                "The final answer is:"
            ]
            
            for prefix in prefixes_to_remove:
                if answer.startswith(prefix):
                    answer = answer[len(prefix):].strip()
                    break
            
            answer = answer.replace('\\n', '\n').replace('\\"', '"').replace("\\'", "'")
            
        except Exception as e:
            answer = f"Error: {str(e)}"
        
        self._save_qa_to_file(question, answer, task_id)
        return answer

    def _save_qa_to_file(self, question: str, answer: str, task_id: str = None):
        """Save question and answer to a results file."""
        try:
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            results_file = "agent_results.json"
            
            if os.path.exists(results_file):
                with open(results_file, 'r', encoding='utf-8') as f:
                    try:
                        results = json.load(f)
                    except:
                        results = []
            else:
                results = []
            
            entry_id = task_id if task_id is not None else len(results) + 1
            
            qa_entry = {
                "id": entry_id,
                "task_id": task_id,
                "timestamp": timestamp,
                "question": question,
                "answer": answer,
                "agent_type": "multimodal_smolagents_gaia"
            }
            
            results.append(qa_entry)
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
                
        except Exception:
            pass


if __name__ == "__main__":
    agent = SmolagentsAgent()