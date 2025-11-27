#  AI Art Classifier - Detector de Arte por IA vs Humana

Sistema completo de classificação de imagens que identifica se uma arte foi criada por humanos ou por Inteligência Artificial. A aplicação consiste em uma API FastAPI com modelo de Deep Learning e um servidor MCP (Model Context Protocol) para integração com Claude Desktop.

##  Índice

- [Visão Geral](#visão-geral)
- [Arquitetura](#arquitetura)
- [Pré-requisitos](#pré-requisitos)
- [Instalação](#instalação)
- [Como Executar](#como-executar)
- [Configuração do MCP no Claude Desktop](#configuração-do-mcp-no-claude-desktop)
- [Como Usar](#como-usar)
- [Endpoints da API](#endpoints-da-api)
- [Resolução de Problemas](#resolução-de-problemas)

---

##  Visão Geral

Este projeto utiliza um modelo de Deep Learning baseado em ResNet50 com uma branch adicional de análise de textura para classificar imagens de arte. O sistema oferece:

- ✅ API REST para classificação de imagens
- ✅ Servidor MCP para integração com Claude Desktop
- ✅ Análise de confiança e probabilidades
- ✅ Suporte a GPU (CUDA) e CPU

##  Arquitetura

```
┌─────────────────────────┐
│   Claude Desktop        │
│   (Interface do Usuário)│
└───────────┬─────────────┘
            │ MCP Protocol
            ▼
┌─────────────────────────┐
│ ai_art_classifier_remote│
│   (Servidor MCP)        │
└───────────┬─────────────┘
            │ HTTP
            ▼
┌─────────────────────────┐
│     server.py           │
│   (API FastAPI)         │
│                         │
│ ResNet50 + TextureBranch│
└─────────────────────────┘
```

---

##  Pré-requisitos

### Software Necessário

1. **Python 3.8+** - [Download aqui](https://www.python.org/downloads/)
2. **Claude Desktop** - [Download aqui](https://claude.ai/download)
3. **Git** (opcional) - Para clonar o repositório

### Dependências Python

As principais bibliotecas necessárias são:
- `torch` e `torchvision` - Framework de Deep Learning
- `fastapi` - Framework web
- `uvicorn` - Servidor ASGI
- `httpx` - Cliente HTTP
- `mcp` (fastmcp) - Protocol para integração com Claude
- `Pillow` - Processamento de imagens

---

##  Instalação

### 1. Clone ou Baixe o Projeto

```bash
# Se estiver usando Git
git clone <seu-repositorio>
cd ti

# Ou simplesmente extraia os arquivos em uma pasta
```

### 2. Instale as Dependências

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install fastapi uvicorn[standard] httpx pillow
pip install fastmcp
```

**Nota:** Se não tiver GPU NVIDIA, instale a versão CPU do PyTorch:
```bash
pip install torch torchvision
```

### 3. Verifique os Arquivos

Certifique-se de que os seguintes arquivos estão na pasta:
-  `server.py` - API FastAPI
-  `ai_art_classifier_remote.py` - Servidor MCP
-  `ai_vs_human_weights.pt` - Pesos do modelo treinado

---

##  Como Executar

### Passo 1: Iniciar a API FastAPI

Abra um terminal na pasta do projeto e execute:

```bash
python server.py
```

Você verá uma saída similar a:

```
Carregando modelo...
✓ Modelo carregado! (Device: cuda)
INFO:     Started server process [xxxxx]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8001
```

** A API está rodando!** Mantenha este terminal aberto.

### Passo 2: Testar a API (Opcional)

Abra o navegador em: `http://localhost:8001`

Você verá informações sobre a API.

---

##  Configuração do MCP no Claude Desktop

Esta é a parte **mais importante** para integrar o classificador com o Claude Desktop.

### Passo 1: Localizar o Arquivo de Configuração

O arquivo de configuração do Claude Desktop está em:

**Windows:**
```
%APPDATA%\Claude\claude_desktop_config.json
```

**Mac:**
```
~/Library/Application Support/Claude/claude_desktop_config.json
```

**Linux:**
```
~/.config/Claude/claude_desktop_config.json
```

### Passo 2: Editar o Arquivo de Configuração

1. **Feche completamente o Claude Desktop** (importante!)

2. Abra o arquivo `claude_desktop_config.json` em um editor de texto

3. Adicione a seguinte configuração:

```json
{
  "mcpServers": {
    "ai-art-classifier": {
      "command": "python",
      "args": [
        "C:\\Users\\dti-\\Desktop\\ti\\ai_art_classifier_remote.py"
      ],
      "env": {}
    }
  }
}
```

** IMPORTANTE:** Ajuste o caminho completo do arquivo `ai_art_classifier_remote.py` de acordo com onde você salvou o projeto!

**Exemplos de caminhos:**
- Windows: `"C:\\Users\\SeuUsuario\\Desktop\\ti\\ai_art_classifier_remote.py"`
- Mac/Linux: `"/home/usuario/projetos/ti/ai_art_classifier_remote.py"`


### Passo 3: Reiniciar o Claude Desktop

1. Salve o arquivo `claude_desktop_config.json`
2. Abra o Claude Desktop
3. Aguarde alguns segundos para o MCP conectar

### Passo 5: Verificar se Funcionou

No Claude Desktop, digite:

```
Você tem acesso à ferramenta classify_art?
```

Se o Claude responder que sim ou mostrar informações sobre a ferramenta, **está funcionando! 🎉**
