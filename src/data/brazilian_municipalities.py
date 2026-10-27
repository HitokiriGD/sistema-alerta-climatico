import csv
from dataclasses import dataclass
import math
from pathlib import Path
import unicodedata


BRAZIL_COUNTRY_CODE = "BR"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MUNICIPALITY_CATALOG_PATH = (
    PROJECT_ROOT / "data" / "reference" / "brazilian_municipalities.csv"
)
CATALOG_COLUMNS = (
    "ibge_code",
    "name",
    "uf",
    "latitude",
    "longitude",
)
MINIMUM_MUNICIPALITY_COUNT = 5_500
BRAZIL_LATITUDE_RANGE = (-35.0, 6.0)
BRAZIL_LONGITUDE_RANGE = (-75.0, -32.0)
STATE_NAMES = {
    "AC": "Acre",
    "AL": "Alagoas",
    "AP": "Amapá",
    "AM": "Amazonas",
    "BA": "Bahia",
    "CE": "Ceará",
    "DF": "Distrito Federal",
    "ES": "Espírito Santo",
    "GO": "Goiás",
    "MA": "Maranhão",
    "MT": "Mato Grosso",
    "MS": "Mato Grosso do Sul",
    "MG": "Minas Gerais",
    "PA": "Pará",
    "PB": "Paraíba",
    "PR": "Paraná",
    "PE": "Pernambuco",
    "PI": "Piauí",
    "RJ": "Rio de Janeiro",
    "RN": "Rio Grande do Norte",
    "RS": "Rio Grande do Sul",
    "RO": "Rondônia",
    "RR": "Roraima",
    "SC": "Santa Catarina",
    "SP": "São Paulo",
    "SE": "Sergipe",
    "TO": "Tocantins",
}
VALID_STATE_CODES = frozenset(STATE_NAMES)


class MunicipalityCatalogError(RuntimeError):
    """Erro controlado ao carregar o catálogo local de municípios."""


class MunicipalityLocationError(RuntimeError):
    """Erro controlado ao validar coordenadas de um município."""


@dataclass(frozen=True)
class BrazilianMunicipality:
    """Identificação e coordenadas de um município brasileiro."""

    ibge_code: int
    name: str
    state_code: str
    state_name: str
    latitude: float
    longitude: float

    @property
    def label(self) -> str:
        return f"{self.name}/{self.state_code}"


def normalize_search_text(value: object) -> str:
    """Normaliza texto para pesquisa sem diferenciar acentos ou maiúsculas."""
    decomposed = unicodedata.normalize("NFD", str(value or ""))
    without_accents = "".join(
        character
        for character in decomposed
        if unicodedata.category(character) != "Mn"
    )
    return " ".join(without_accents.casefold().split())


def format_municipality_option(municipality: BrazilianMunicipality) -> str:
    """Mantém o rótulo visual e torna a busca nativa tolerante a acentos."""
    return unicodedata.normalize("NFD", municipality.label)


def filter_municipalities(
    municipalities: tuple[BrazilianMunicipality, ...],
    query: str,
) -> tuple[BrazilianMunicipality, ...]:
    """Filtra municípios pelo nome ou pela UF sem diferenciar acentos."""
    normalized_query = normalize_search_text(query)
    if not normalized_query:
        return municipalities
    return tuple(
        municipality
        for municipality in municipalities
        if normalized_query in normalize_search_text(municipality.label)
    )


def find_municipality_by_name(
    municipalities: tuple[BrazilianMunicipality, ...],
    name: str,
    state_code: str | None = None,
) -> BrazilianMunicipality | None:
    """Localiza uma opção do catálogo por nome e, opcionalmente, por UF."""
    normalized_name = normalize_search_text(name)
    normalized_state = str(state_code or "").strip().upper()
    matches = [
        municipality
        for municipality in municipalities
        if normalize_search_text(municipality.name) == normalized_name
        and (
            not normalized_state
            or municipality.state_code.upper() == normalized_state
        )
    ]
    if len(matches) != 1:
        return None
    return matches[0]


def load_local_municipality_catalog(
    catalog_path: str | Path = DEFAULT_MUNICIPALITY_CATALOG_PATH,
) -> tuple[BrazilianMunicipality, ...]:
    """Carrega e valida o catálogo municipal versionado no projeto."""
    path = Path(catalog_path)
    if not path.is_file():
        raise MunicipalityCatalogError(
            "O catálogo local de municípios não foi encontrado em "
            f"'{path}'. Execute scripts/build_municipality_catalog.py."
        )

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as catalog_file:
            reader = csv.DictReader(catalog_file)
            missing_columns = set(CATALOG_COLUMNS) - set(reader.fieldnames or ())
            if missing_columns:
                missing = ", ".join(sorted(missing_columns))
                raise MunicipalityCatalogError(
                    f"O catálogo local não contém as colunas obrigatórias: {missing}."
                )
            municipalities = [
                _parse_catalog_row(row, row_number)
                for row_number, row in enumerate(reader, start=2)
            ]
    except MunicipalityCatalogError:
        raise
    except (OSError, UnicodeError, csv.Error) as error:
        raise MunicipalityCatalogError(
            "Não foi possível ler o catálogo local de municípios."
        ) from error

    _validate_catalog(municipalities)
    return tuple(
        sorted(
            municipalities,
            key=lambda municipality: (
                normalize_search_text(municipality.name),
                municipality.state_code,
            ),
        )
    )


def validate_municipality_coordinates(
    latitude: object,
    longitude: object,
) -> tuple[float, float]:
    """Valida coordenadas finitas dentro do envelope geográfico do Brasil."""
    try:
        parsed_latitude = float(latitude)
        parsed_longitude = float(longitude)
    except (TypeError, ValueError) as error:
        raise MunicipalityLocationError(
            "O município selecionado não possui coordenadas locais válidas."
        ) from error

    if not _coordinates_are_valid(parsed_latitude, parsed_longitude):
        raise MunicipalityLocationError(
            "O município selecionado não possui coordenadas locais válidas."
        )
    return parsed_latitude, parsed_longitude


def _parse_catalog_row(
    row: dict[str, str | None],
    row_number: int,
) -> BrazilianMunicipality:
    try:
        ibge_code = int(str(row.get("ibge_code") or "").strip())
        latitude = float(str(row.get("latitude") or "").strip())
        longitude = float(str(row.get("longitude") or "").strip())
    except ValueError as error:
        raise MunicipalityCatalogError(
            f"O catálogo local contém valores inválidos na linha {row_number}."
        ) from error

    name = str(row.get("name") or "").strip()
    state_code = str(row.get("uf") or "").strip().upper()
    if len(str(ibge_code)) != 7 or ibge_code <= 0:
        raise MunicipalityCatalogError(
            f"O catálogo local contém código IBGE inválido na linha {row_number}."
        )
    if not name:
        raise MunicipalityCatalogError(
            f"O catálogo local contém nome vazio na linha {row_number}."
        )
    if state_code not in VALID_STATE_CODES:
        raise MunicipalityCatalogError(
            f"O catálogo local contém UF inválida na linha {row_number}."
        )
    if not _coordinates_are_valid(latitude, longitude):
        raise MunicipalityCatalogError(
            f"O catálogo local contém coordenadas inválidas na linha {row_number}."
        )
    return BrazilianMunicipality(
        ibge_code=ibge_code,
        name=name,
        state_code=state_code,
        state_name=STATE_NAMES[state_code],
        latitude=latitude,
        longitude=longitude,
    )


def _validate_catalog(municipalities: list[BrazilianMunicipality]) -> None:
    if len(municipalities) < MINIMUM_MUNICIPALITY_COUNT:
        raise MunicipalityCatalogError(
            "O catálogo local está incompleto: foram encontrados apenas "
            f"{len(municipalities)} municípios."
        )

    codes = [municipality.ibge_code for municipality in municipalities]
    if len(codes) != len(set(codes)):
        raise MunicipalityCatalogError(
            "O catálogo local contém códigos IBGE duplicados."
        )

    names_by_state = [
        (normalize_search_text(municipality.name), municipality.state_code)
        for municipality in municipalities
    ]
    if len(names_by_state) != len(set(names_by_state)):
        raise MunicipalityCatalogError(
            "O catálogo local contém municípios duplicados na mesma UF."
        )


def _coordinates_are_valid(latitude: float, longitude: float) -> bool:
    return (
        math.isfinite(latitude)
        and math.isfinite(longitude)
        and BRAZIL_LATITUDE_RANGE[0] <= latitude <= BRAZIL_LATITUDE_RANGE[1]
        and BRAZIL_LONGITUDE_RANGE[0] <= longitude <= BRAZIL_LONGITUDE_RANGE[1]
    )
