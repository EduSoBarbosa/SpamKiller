from pathlib import Path
import cloudpickle
import joblib

PASTA = Path(__file__).resolve().parent


def carregar(configuracao="precisao"):
    if configuracao not in {"precisao", "recall"}:
        raise ValueError("Use 'precisao' ou 'recall'.")

    pacote = joblib.load(PASTA / f"spam_{configuracao}.joblib")

    with (PASTA / "preprocessamento.pkl").open("rb") as arquivo:
        funcoes = cloudpickle.load(arquivo)

    return pacote, funcoes


def prever_texto_processado(texto, pacote):
    if not texto.strip():
        raise ValueError("O texto ficou vazio após o processamento.")

    pipeline = pacote["pipeline"]
    indice = list(pipeline.classes_).index(pacote["classe_positiva"])

    probabilidade = float(
        pipeline.predict_proba([texto])[0, indice]
    )

    classe = int(probabilidade >= pacote["limiar"])

    return {
        "classe": classe,
        "rotulo": pacote["rotulos"][classe],
        "probabilidade_spam": probabilidade,
        "limiar": pacote["limiar"],
        "configuracao": pacote["configuracao"]
    }


def classificar(assunto, corpo, pacote, funcoes):
    texto = (
        f"{funcoes['limpar_assunto'](assunto)} "
        f"{funcoes['limpar_texto'](corpo)}"
    ).strip()

    return prever_texto_processado(texto, pacote)


def classificar_arquivo(caminho, pacote, funcoes):
    texto = funcoes["processar_email"](caminho)
    return prever_texto_processado(texto, pacote)


if __name__ == "__main__":
    pacote, funcoes = carregar("precisao")

    resultado = classificar(
        assunto="Meeting tomorrow",
        corpo="Hi team, our project meeting is tomorrow at 10 AM.",
        pacote=pacote,
        funcoes=funcoes
    )

    print(resultado)
