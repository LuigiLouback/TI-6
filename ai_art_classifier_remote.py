from typing import Any
import sys
import httpx
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
import os

load_dotenv()
API_URL = os.getenv("API_URL", "http://localhost:8001")
mcp = FastMCP("ai-art-classifier-remote")


def image_to_base64(image_path: str) -> bytes:
  """Converte imagem local para bytes"""
  with open(image_path, "rb") as f:
      return f.read()


@mcp.tool()
def classify_art(image_path: str) -> dict[str, Any]:
  """
  Classifica uma imagem de arte como criada por humano ou por IA.
  
  Args:
      image_path: Caminho local da imagem
  
  Returns:
      Classificação, confiança e probabilidades
  """
  try:
      print(f"[DEBUG] Enviando para API: {image_path}", file=sys.stderr)
      
      with open(image_path, "rb") as f:
          image_bytes = f.read()
 
      files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
      
      response = httpx.post(
          f"{API_URL}/classify",
          files=files,
          timeout=30.0
      )
      
      response.raise_for_status()
      result = response.json()
      
      print(f"[DEBUG] Resposta da API: {result}", file=sys.stderr)
      return result
      
  except FileNotFoundError:
      return {
          "success": False,
          "error": f"Arquivo não encontrado: {image_path}"
      }
  except httpx.HTTPError as e:
      return {
          "success": False,
          "error": f"Erro na requisição: {str(e)}"
      }
  except Exception as e:
      return {
          "success": False,
          "error": str(e)
      }

@mcp.tool()
def get_model_info() -> dict[str, str]:
  """Retorna informações sobre o modelo remoto."""
  try:
      response = httpx.get(f"{API_URL}/health", timeout=30.0)
      return response.json()
  except:
      return {"status": "offline", "error": "Não foi possível conectar ao servidor"}

if __name__ == "__main__":
  print("="*60, file=sys.stderr)
  print("AI Art Classifier MCP Server (Remote)", file=sys.stderr)
  print(f"API: {API_URL}", file=sys.stderr)
  print("="*60, file=sys.stderr)
  
  mcp.run(transport='stdio')