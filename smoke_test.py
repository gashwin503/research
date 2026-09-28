from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from dotenv import load_dotenv

load_dotenv()

endpoint = os.getenv("ENDPOINT")

# Format: "https://resource_name.services.ai.azure.com/api/projects/project_name"
FOUNDRY_PROJECT_ENDPOINT = endpoint

# Create project and openai clients to call Foundry API
project = AIProjectClient(
    endpoint=FOUNDRY_PROJECT_ENDPOINT,
    credential=DefaultAzureCredential(),
)
openai = project.get_openai_client()

# Run a responses API call
response = openai.responses.create(
    model="gpt-5-mini",  # supports all Foundry direct models
    input="What is the size of France in square miles?",
)
if not response.output_text or not response.output_text.strip():
    raise RuntimeError("Response output text was empty.")

print(f"Response output: {response.output_text}")
