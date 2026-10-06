| Função | Responsabilidade |
| --- | --- |
| limpar_texto | Substitui URLs por URL e endereços por EMAIL; normaliza espaços |
| limpar_assunto | Remove marcações explícitas de spam previstas na regex; chama limpar_texto |
| extrair_corpo | Percorre MIME; extrai texto simples e HTML convertido; ignora partes marcadas como anexo |
| processar_email | Lê bytes com BytesParser; une assunto e corpo tratados |
| carregar_pasta | Ignora cmds e diretórios; associa o texto não vazio ao rótulo da pasta |
