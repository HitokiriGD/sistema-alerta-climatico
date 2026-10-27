import argparse
import csv
import io
import json
from pathlib import Path
import sys
from typing import Any

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.brazilian_municipalities import CATALOG_COLUMNS
from src.data.brazilian_municipalities import DEFAULT_MUNICIPALITY_CATALOG_PATH
from src.data.brazilian_municipalities import MunicipalityCatalogError
from src.data.brazilian_municipalities import load_local_municipality_catalog
from src.data.brazilian_municipalities import normalize_search_text


IBGE_MUNICIPALITIES_URL = (
    "https://servicodados.ibge.gov.br/api/v1/localidades/municipios"
)
KELVINS_MUNICIPALITIES_URL = (
    "https://raw.githubusercontent.com/kelvins/municipios-brasileiros/"
    "main/csv/municipios.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Atualiza o catálogo local usando nomes, códigos e UFs do IBGE e "
            "coordenadas auxiliares de kelvins/municipios-brasileiros."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_MUNICIPALITY_CATALOG_PATH,
        help="Caminho do CSV gerado.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Tempo limite, em segundos, para cada fonte.",
    )
    return parser.parse_args()


def download_json(url: str, timeout: float) -> object:
    response = requests.get(url, params={"orderBy": "nome"}, timeout=timeout)
    response.raise_for_status()
    return json.loads(response.content.decode("utf-8-sig"))


def download_text(url: str, timeout: float) -> str:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.content.decode("utf-8-sig")


def build_catalog_rows(
    ibge_payload: object,
    coordinates_csv: str,
) -> list[dict[str, object]]:
    """Cruza fontes por código IBGE e mantém a identificação oficial."""
    if not isinstance(ibge_payload, list):
        raise ValueError("A fonte do IBGE não retornou uma lista de municípios.")

    coordinates_by_code: dict[int, tuple[float, float]] = {}
    for row in csv.DictReader(io.StringIO(coordinates_csv)):
        try:
            ibge_code = int(row["codigo_ibge"])
            coordinates = (float(row["latitude"]), float(row["longitude"]))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "A base de coordenadas contém um registro inválido."
            ) from error
        if ibge_code in coordinates_by_code:
            raise ValueError(
                f"A base de coordenadas repete o código IBGE {ibge_code}."
            )
        coordinates_by_code[ibge_code] = coordinates

    rows: list[dict[str, object]] = []
    official_codes: set[int] = set()
    for item in ibge_payload:
        official = _parse_ibge_municipality(item)
        ibge_code = int(official["ibge_code"])
        if ibge_code in official_codes:
            raise ValueError(f"O IBGE repetiu o código {ibge_code}.")
        official_codes.add(ibge_code)
        if ibge_code not in coordinates_by_code:
            raise ValueError(
                "A base de coordenadas não contém o município oficial de código "
                f"{ibge_code}."
            )
        latitude, longitude = coordinates_by_code[ibge_code]
        rows.append(
            {
                **official,
                "latitude": latitude,
                "longitude": longitude,
            }
        )

    extra_codes = set(coordinates_by_code) - official_codes
    if extra_codes:
        examples = ", ".join(str(code) for code in sorted(extra_codes)[:5])
        raise ValueError(
            "A base de coordenadas contém códigos ausentes do cadastro atual do "
            f"IBGE: {examples}."
        )
    return sorted(
        rows,
        key=lambda row: (normalize_search_text(row["name"]), str(row["uf"])),
    )


def write_catalog(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(
                output_file,
                fieldnames=CATALOG_COLUMNS,
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
        load_local_municipality_catalog(temporary_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    temporary_path.replace(output_path)


def _parse_ibge_municipality(item: object) -> dict[str, object]:
    if not isinstance(item, dict):
        raise ValueError("O IBGE retornou um registro municipal inválido.")
    state = _extract_state(item)
    try:
        ibge_code = int(item["id"])
        name = str(item["nome"]).strip()
        state_code = str(state["sigla"]).strip().upper()
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("O IBGE retornou um registro municipal incompleto.") from error
    if not name:
        raise ValueError(f"O município {ibge_code} não possui nome oficial.")
    return {"ibge_code": ibge_code, "name": name, "uf": state_code}


def _extract_state(item: dict[str, Any]) -> dict[str, Any]:
    immediate_region = item.get("regiao-imediata")
    if isinstance(immediate_region, dict):
        intermediate_region = immediate_region.get("regiao-intermediaria")
        if isinstance(intermediate_region, dict):
            state = intermediate_region.get("UF")
            if isinstance(state, dict):
                return state

    microregion = item.get("microrregiao")
    if isinstance(microregion, dict):
        mesoregion = microregion.get("mesorregiao")
        if isinstance(mesoregion, dict):
            state = mesoregion.get("UF")
            if isinstance(state, dict):
                return state
    raise ValueError("O IBGE retornou um município sem UF identificável.")


def main() -> int:
    args = parse_args()
    try:
        ibge_payload = download_json(IBGE_MUNICIPALITIES_URL, args.timeout)
        coordinates_csv = download_text(KELVINS_MUNICIPALITIES_URL, args.timeout)
        rows = build_catalog_rows(ibge_payload, coordinates_csv)
        write_catalog(rows, args.output)
        municipalities = load_local_municipality_catalog(args.output)
    except (
        MunicipalityCatalogError,
        OSError,
        ValueError,
        requests.RequestException,
    ) as error:
        print(f"Falha ao atualizar o catálogo: {error}", file=sys.stderr)
        return 1

    print(
        f"Catálogo atualizado em {args.output} com "
        f"{len(municipalities)} municípios."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
