from typing import Any
import sys
import httpx
import base64
from mcp.server.fastmcp import FastMCP

API_URL = "http://localhost:8001"  

mcp = FastMCP("ai-art-classifier-remote")


@mcp.tool()
def classify_art_from_path(image_path: str) -> dict[str, Any]:
  """
  Classifica uma imagem de arte como criada por humano ou por IA a partir de um caminho local.
  
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
def classify_art_from_base64(image_base64: str, filename: str = "image.jpg") -> dict[str, Any]:
  """
  Classifica uma imagem de arte como criada por humano ou por IA a partir de dados base64.
  
  Args:
      image_base64: Imagem codificada em base64 (string)
      filename: Nome do arquivo (opcional, padrão: image.jpg)
  
  Returns:
      Classificação, confiança e probabilidades
  """
  try:
      print(f"[DEBUG] Decodificando imagem base64", file=sys.stderr)
      
      
      if ',' in image_base64 and image_base64.startswith('data:'):
          image_base64 = image_base64.split(',', 1)[1]
      
      
      image_bytes = base64.b64decode(image_base64)
      
      print(f"[DEBUG] Enviando {len(image_bytes)} bytes para API", file=sys.stderr)
      
     
      mime_type = "image/jpeg"
      if filename.lower().endswith('.png'):
          mime_type = "image/png"
      elif filename.lower().endswith('.webp'):
          mime_type = "image/webp"
      
      files = {"file": (filename, image_bytes, mime_type)}
      
      response = httpx.post(
          f"{API_URL}/classify",
          files=files,
          timeout=30.0
      )
      
      response.raise_for_status()
      result = response.json()
      
      print(f"[DEBUG] Resposta da API: {result}", file=sys.stderr)
      return result
      
  except base64.binascii.Error:
      return {
          "success": False,
          "error": "Dados base64 inválidos"
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
def classify_art_from_url(image_url: str) -> dict[str, Any]:
  """
  Classifica uma imagem de arte a partir de uma URL.
  
  Args:
      image_url: URL pública da imagem (http:// ou https://)
  
  Returns:
      Classificação, confiança e probabilidades
  """
  try:
      print(f"[DEBUG] Baixando imagem de: {image_url}", file=sys.stderr)
      
      img_response = httpx.get(image_url, timeout=30.0, follow_redirects=True)
      img_response.raise_for_status()
      image_bytes = img_response.content
      
      print(f"[DEBUG] Imagem baixada: {len(image_bytes)} bytes", file=sys.stderr)
      
      filename = "image.jpg"
      if image_url.lower().endswith('.png'):
          filename = "image.png"
          mime_type = "image/png"
      elif image_url.lower().endswith('.webp'):
          filename = "image.webp"
          mime_type = "image/webp"
      elif image_url.lower().endswith(('.jpg', '.jpeg')):
          mime_type = "image/jpeg"
      else:
          mime_type = "image/jpeg"
      
      files = {"file": (filename, image_bytes, mime_type)}
      
      response = httpx.post(
          f"{API_URL}/classify",
          files=files,
          timeout=30.0
      )
      
      response.raise_for_status()
      result = response.json()
      
      print(f"[DEBUG] Resposta da API: {result}", file=sys.stderr)
      return result
      
  except httpx.HTTPError as e:
      return {
          "success": False,
          "error": f"Erro ao baixar ou processar imagem: {str(e)}"
      }
  except Exception as e:
      return {
          "success": False,
          "error": str(e)
      }


@mcp.tool()
def classify_art_from_bytes(image_hex: str, filename: str = "image.jpg") -> dict[str, Any]:
  """
  Classifica uma imagem a partir de bytes em formato hexadecimal.
  
  Args:
      image_hex: Bytes da imagem em formato hexadecimal (string hex)
      filename: Nome do arquivo (opcional, padrão: image.jpg)
  
  Returns:
      Classificação, confiança e probabilidades
  """
  try:
      print(f"[DEBUG] Convertendo hex para bytes", file=sys.stderr)
      
      image_bytes = bytes.fromhex(image_hex)
      
      print(f"[DEBUG] Enviando {len(image_bytes)} bytes para API", file=sys.stderr)
      
      mime_type = "image/jpeg"
      if filename.lower().endswith('.png'):
          mime_type = "image/png"
      elif filename.lower().endswith('.webp'):
          mime_type = "image/webp"
      
      files = {"file": (filename, image_bytes, mime_type)}
      
      response = httpx.post(
          f"{API_URL}/classify",
          files=files,
          timeout=30.0
      )
      
      response.raise_for_status()
      result = response.json()
      
      print(f"[DEBUG] Resposta da API: {result}", file=sys.stderr)
      return result
      
  except ValueError as e:
      return {
          "success": False,
          "error": f"Formato hexadecimal inválido: {str(e)}"
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
  """Retorna se o modelo esta funcionando"""
  try:
      response = httpx.get(f"{API_URL}/health", timeout=30.0)
      return response.json()
  except:
      return {"status": "offline", "error": "Não foi possível conectar ao servidor"}


@mcp.tool()
def get_model_specifications() -> dict[str, Any]:
  """
  Retorna especificações completas do modelo de classificação.
  
  Inclui:
  - Arquitetura do modelo
  - Dispositivo (CPU/GPU)
  - Endpoints disponíveis
  - Status do servidor
  
  Returns:
      Especificações detalhadas do modelo
  """
  try:
      print("[DEBUG] Buscando especificações do modelo", file=sys.stderr)
      
      response = httpx.get(f"{API_URL}/", timeout=30.0)
      response.raise_for_status()
      
      specs = response.json()
      print(f"[DEBUG] Especificações obtidas: {specs}", file=sys.stderr)
      
      return {
          "success": True,
          "model_name": specs.get("model", "N/A"),
          "device": specs.get("device", "N/A"),
          "status": specs.get("status", "unknown"),
          "endpoints": specs.get("endpoints", {}),
          "api_url": API_URL
      }
      
  except httpx.HTTPError as e:
      return {
          "success": False,
          "error": f"Erro ao conectar ao servidor: {str(e)}",
          "api_url": API_URL
      }
  except Exception as e:
      return {
          "success": False,
          "error": f"Erro inesperado: {str(e)}",
          "api_url": API_URL
      }

if __name__ == "__main__":
  print("="*60, file=sys.stderr)
  print("AI Art Classifier MCP Server (Remote)", file=sys.stderr)
  print(f"API: {API_URL}", file=sys.stderr)
  print("="*60, file=sys.stderr)
  
  mcp.run(transport='stdio')