def ler_pdf(caminho):
    """Lê arquivos PDF com tratamento melhorado"""
    import PyPDF2
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
    import docx
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
    """Busca múltiplas palavras ou frases no texto"""
    resultados = {}
    texto_lower = texto.lower()

    # Remove pontuação para busca mais abrangente
    import string
    texto_sem_pontuacao = texto.translate(str.maketrans('', '', string.punctuation))
    texto_sem_pontuacao_lower = texto_sem_pontuacao.lower()

    for palavra in palavras:
        palavra = palavra.strip()
        if not palavra:
            continue

        palavra_lower = palavra.lower()
        palavra_sem_pontuacao = palavra.translate(str.maketrans('', '', string.punctuation)).lower()

        encontradas = []
        start = 0

        # Busca tanto a palavra original quanto sem pontuação
        for termo_busca in [palavra_lower, palavra_sem_pontuacao]:
            if not termo_busca:
                continue

            while True:
                pos = texto_sem_pontuacao_lower.find(termo_busca, start) if termo_busca == palavra_sem_pontuacao else texto_lower.find(termo_busca, start)
                if pos == -1:
                    break

                # Extrai contexto
                inicio = max(0, pos - 30)
                fim = min(len(texto), pos + len(termo_busca) + 30)
                contexto = texto[inicio:fim].replace('\n', ' ')

                # Remove pontuação do contexto para exibição
                contexto_sem_pontuacao = contexto.translate(str.maketrans('', '', string.punctuation))

                if inicio > 0:
                    contexto_sem_pontuacao = "..." + contexto_sem_pontuacao
                if fim < len(texto):
                    contexto_sem_pontuacao = contexto_sem_pontuacao + "..."

                linha = texto.count('\n', 0, pos) + 1
                encontradas.append(f"Linha {linha}: {contexto_sem_pontuacao.strip()}")
                start = pos + len(termo_busca)

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

    # Pede as palavras ou frases para buscar
    palavras_input = input("Digite as palavras ou frases a buscar (separadas por vírgula): ").strip()
    if not palavras_input:
        print("Nenhuma palavra ou frase fornecida!")
        return

    palavras = [p.strip() for p in palavras_input.split(",") if p.strip()]

    resultados_por_arquivo = {}

    for caminho in lista_caminhos:
        # Verifica extensão do arquivo
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

        # Realiza a busca
        resultados = buscar_palavras(texto, palavras)
        resultados_por_arquivo[caminho] = resultados

    # Avalia quais arquivos têm mais ocorrências
    relevancia = avaliar_relevancia(resultados_por_arquivo)

    print("\n" + "="*50)
    print("\n📊 RESULTADOS GERAIS (ordenados por relevância):")
    for arquivo, total in relevancia:
        print(f"{arquivo}: {total} ocorrências no total")

    # Mostra resultados detalhados para cada arquivo
    for arquivo, resultados in resultados_por_arquivo.items():
        print("\n" + "="*50)
        print(f"\n📂 ARQUIVO: {arquivo}")

        for palavra, ocorrencias in resultados.items():
            print(f"\n🔍 Resultados para '{palavra}':")
            if ocorrencias:
                for i, ocorrencia in enumerate(ocorrencias[:5], 1):  # Mostra até 5 ocorrências por palavra/frase
                    print(f"{i}. {ocorrencia}")
                if len(ocorrencias) > 5:
                    print(f"... e mais {len(ocorrencias)-5} ocorrências (total: {len(ocorrencias)})")
            else:
                print(f"Nenhuma ocorrência encontrada para '{palavra}'")

    print("\n" + "="*50)
    print("\nOs melhores arquivos são:")
    for i, (arquivo, total) in enumerate(relevancia[:3], 1):  # Mostra os top 3
        if total > 0:
            print(f"{i}. {arquivo} ({total} ocorrências)")

if __name__ == "__main__":
    main()''
