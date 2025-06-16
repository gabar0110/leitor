import PyPDF2
import docx
import string
import re

def ler_pdf(caminho):
    """Lê arquivos PDF com tratamento melhorado"""
    texto = ""
    try:
        with open(caminho, 'rb') as f:
            leitor = PyPDF2.PdfReader(f)
            for pagina in leitor.pages:
                texto += pagina.extract_text() + "\n"
    except Exception as e:
        print(f"Erro ao ler PDF {caminho}: {e}")
    return texto

def ler_docx(caminho):
    """Lê arquivos DOCX"""
    try:
        doc = docx.Document(caminho)
        return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    except Exception as e:
        print(f"Erro ao ler DOCX {caminho}: {e}")
        return ""

def ler_txt(caminho):
    """Lê arquivos TXT com codificação correta"""
    try:
        with open(caminho, 'r', encoding='utf-8') as f:
            return f.read()
    except:
        try:
            with open(caminho, 'r', encoding='latin-1') as f:
                return f.read()
        except Exception as e:
            print(f"Erro ao ler TXT {caminho}: {e}")
            return ""

def buscar_palavras(texto, palavras):
    """Busca múltiplas palavras ou frases no texto considerando palavras completas"""
    resultados = {}

    for palavra in palavras:
        palavra = palavra.strip()
        if not palavra:
            continue

        encontradas = []

        # Para palavras únicas (sem espaços)
        if ' ' not in palavra:
            # Remove pontuação da palavra de busca
            palavra_limpa = palavra.translate(str.maketrans('', '', string.punctuation)).lower()

            # Usa regex para encontrar apenas palavras completas
            padrao = re.compile(r'\b' + re.escape(palavra_limpa) + r'\b', re.IGNORECASE)

            for match in padrao.finditer(texto):
                pos = match.start()

                # Extrai contexto
                inicio = max(0, pos - 30)
                fim = min(len(texto), pos + len(palavra_limpa) + 30)
                contexto = texto[inicio:fim].replace('\n', ' ')

                if inicio > 0:
                    contexto = "..." + contexto
                if fim < len(texto):
                    contexto = contexto + "..."

                linha = texto.count('\n', 0, pos) + 1
                encontradas.append(f"Linha {linha}: {contexto.strip()}")

        # Para frases (com espaços)
        else:
            frase_lower = palavra.lower()
            start = 0

            while True:
                pos = texto.lower().find(frase_lower, start)
                if pos == -1:
                    break

                # Verifica se é uma correspondência exata (não parte de outra palavra)
                inicio_frase = pos
                fim_frase = pos + len(frase_lower)

                # Verifica os caracteres antes e depois
                antes_ok = (inicio_frase == 0 or
                           texto[inicio_frase-1] in string.whitespace + string.punctuation)
                depois_ok = (fim_frase == len(texto) or
                            texto[fim_frase] in string.whitespace + string.punctuation)

                if antes_ok and depois_ok:
                    # Extrai contexto
                    inicio = max(0, pos - 30)
                    fim = min(len(texto), pos + len(frase_lower) + 30)
                    contexto = texto[inicio:fim].replace('\n', ' ')

                    if inicio > 0:
                        contexto = "..." + contexto
                    if fim < len(texto):
                        contexto = contexto + "..."

                    linha = texto.count('\n', 0, pos) + 1
                    encontradas.append(f"Linha {linha}: {contexto.strip()}")

                start = pos + 1

        # Remove duplicados mantendo a ordem
        seen = set()
        encontradas = [x for x in encontradas if not (x in seen or seen.add(x))]

        resultados[palavra] = encontradas

    return resultados

def avaliar_relevancia(resultados):
    """Avalia quais arquivos têm mais ocorrências"""
    relevancia = {}
    for arquivo, palavras in resultados.items():
        total = sum(len(ocorrencias) for ocorrencias in palavras.values())
        relevancia[arquivo] = total
    return sorted(relevancia.items(), key=lambda x: x[1], reverse=True)

def main():
    print("\n=== BUSCADOR MULTI-PALAVRAS ===")
    print("Encontre várias palavras ou frases de uma vez em múltiplos arquivos\n")

    caminhos = input("Caminhos completos dos arquivos (separados por vírgula): ").strip('"').strip()
    if not caminhos:
        print("Caminhos inválidos!")
        return

    lista_caminhos = [c.strip().strip('"') for c in caminhos.split(",") if c.strip()]

    palavras_input = input("Digite as palavras ou frases a buscar (separadas por vírgula): ").strip()
    if not palavras_input:
        print("Nenhuma palavra ou frase fornecida!")
        return

    palavras = [p.strip() for p in palavras_input.split(",") if p.strip()]

    resultados_por_arquivo = {}

    for caminho in lista_caminhos:
        if caminho.lower().endswith('.pdf'):
            texto = ler_pdf(caminho)
        elif caminho.lower().endswith('.docx'):
            texto = ler_docx(caminho)
        elif caminho.lower().endswith('.txt'):
            texto = ler_txt(caminho)
        else:
            print(f"Formato não suportado para {caminho}. Use PDF, DOCX ou TXT.")
            continue

        if not texto:
            print(f"Não foi possível ler o arquivo {caminho} ou o arquivo está vazio.")
            continue

        resultados = buscar_palavras(texto, palavras)
        resultados_por_arquivo[caminho] = resultados

    relevancia = avaliar_relevancia(resultados_por_arquivo)

    print("\n" + "="*50)
    print("\n📊 RESULTADOS GERAIS (ordenados por relevância):")
    for arquivo, total in relevancia:
        print(f"{arquivo}: {total} ocorrências no total")

    for arquivo, resultados in resultados_por_arquivo.items():
        print("\n" + "="*50)
        print(f"\n📂 ARQUIVO: {arquivo}")

        for palavra, ocorrencias in resultados.items():
            print(f"\n🔍 Resultados para '{palavra}':")
            if ocorrencias:
                for i, ocorrencia in enumerate(ocorrencias[:5], 1):
                    print(f"{i}. {ocorrencia}")
                if len(ocorrencias) > 5:
                    print(f"... e mais {len(ocorrencias)-5} ocorrências (total: {len(ocorrencias)})")
            else:
                print(f"Nenhuma ocorrência encontrada para '{palavra}'")

    print("\n" + "="*50)
    print("\nOs melhores arquivos são:")
    for i, (arquivo, total) in enumerate(relevancia[:3], 1):
        if total > 0:
            print(f"{i}. {arquivo} ({total} ocorrências)")

if __name__ == "__main__":
    main()
