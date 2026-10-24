from dataclasses import dataclass
import math
import unicodedata
from typing import Any

import requests


IBGE_MUNICIPALITIES_URL = (
    "https://servicodados.ibge.gov.br/api/v1/localidades/municipios"
)
IBGE_MUNICIPALITY_MESH_URL = (
    "https://servicodados.ibge.gov.br/api/v3/malhas/municipios/{ibge_code}"
)
BRAZIL_COUNTRY_CODE = "BR"


class MunicipalityCatalogError(RuntimeError):
    """Erro controlado ao carregar o catálogo oficial de municípios."""


class MunicipalityLocationError(RuntimeError):
    """Erro controlado ao resolver uma coordenada segura para o município."""


@dataclass(frozen=True)
class BrazilianMunicipality:
    """Identificação oficial de um município brasileiro."""

    ibge_code: int
    name: str
    state_code: str
    state_name: str

    @property
    def label(self) -> str:
        return f"{self.name}/{self.state_code}"


FALLBACK_CAPITALS: tuple[BrazilianMunicipality, ...] = (
    BrazilianMunicipality(0, "Rio Branco", "AC", "Acre"),
    BrazilianMunicipality(0, "Maceió", "AL", "Alagoas"),
    BrazilianMunicipality(0, "Macapá", "AP", "Amapá"),
    BrazilianMunicipality(0, "Manaus", "AM", "Amazonas"),
    BrazilianMunicipality(0, "Salvador", "BA", "Bahia"),
    BrazilianMunicipality(0, "Fortaleza", "CE", "Ceará"),
    BrazilianMunicipality(0, "Brasília", "DF", "Distrito Federal"),
    BrazilianMunicipality(0, "Vitória", "ES", "Espírito Santo"),
    BrazilianMunicipality(0, "Goiânia", "GO", "Goiás"),
    BrazilianMunicipality(0, "São Luís", "MA", "Maranhão"),
    BrazilianMunicipality(0, "Cuiabá", "MT", "Mato Grosso"),
    BrazilianMunicipality(0, "Campo Grande", "MS", "Mato Grosso do Sul"),
    BrazilianMunicipality(0, "Belo Horizonte", "MG", "Minas Gerais"),
    BrazilianMunicipality(0, "Belém", "PA", "Pará"),
    BrazilianMunicipality(0, "João Pessoa", "PB", "Paraíba"),
    BrazilianMunicipality(0, "Curitiba", "PR", "Paraná"),
    BrazilianMunicipality(0, "Recife", "PE", "Pernambuco"),
    BrazilianMunicipality(0, "Teresina", "PI", "Piauí"),
    BrazilianMunicipality(0, "Rio de Janeiro", "RJ", "Rio de Janeiro"),
    BrazilianMunicipality(0, "Natal", "RN", "Rio Grande do Norte"),
    BrazilianMunicipality(0, "Porto Alegre", "RS", "Rio Grande do Sul"),
    BrazilianMunicipality(0, "Porto Velho", "RO", "Rondônia"),
    BrazilianMunicipality(0, "Boa Vista", "RR", "Roraima"),
    BrazilianMunicipality(0, "Florianópolis", "SC", "Santa Catarina"),
    BrazilianMunicipality(0, "São Paulo", "SP", "São Paulo"),
    BrazilianMunicipality(0, "Aracaju", "SE", "Sergipe"),
    BrazilianMunicipality(0, "Palmas", "TO", "Tocantins"),
)


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


def fetch_ibge_municipalities(
    request_get: Any = requests.get,
    timeout: float = 20.0,
) -> tuple[BrazilianMunicipality, ...]:
    """Carrega o catálogo de municípios diretamente da API do IBGE."""
    try:
        response = request_get(
            IBGE_MUNICIPALITIES_URL,
            params={"orderBy": "nome"},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as error:
        raise MunicipalityCatalogError(
            "Não foi possível carregar o catálogo de municípios do IBGE."
        ) from error

    if not isinstance(payload, list):
        raise MunicipalityCatalogError(
            "O catálogo de municípios do IBGE retornou um formato inesperado."
        )

    municipalities: list[BrazilianMunicipality] = []
    for item in payload:
        municipality = _parse_municipality(item)
        if municipality is not None:
            municipalities.append(municipality)

    unique_by_code = {
        municipality.ibge_code: municipality for municipality in municipalities
    }
    if len(unique_by_code) < 5_000:
        raise MunicipalityCatalogError(
            "O catálogo de municípios do IBGE está incompleto no momento."
        )

    return tuple(
        sorted(
            unique_by_code.values(),
            key=lambda municipality: (
                normalize_search_text(municipality.name),
                municipality.state_code,
            ),
        )
    )


def fetch_ibge_municipality_coordinates(
    ibge_code: int,
    request_get: Any = requests.get,
    timeout: float = 20.0,
) -> tuple[float, float]:
    """Obtém um ponto interno da malha oficial do município no IBGE."""
    if int(ibge_code) <= 0:
        raise MunicipalityLocationError(
            "O município selecionado não possui código IBGE validado."
        )

    try:
        response = request_get(
            IBGE_MUNICIPALITY_MESH_URL.format(ibge_code=int(ibge_code)),
            params={
                "qualidade": "minima",
                "formato": "application/vnd.geo+json",
            },
            headers={"Accept": "application/vnd.geo+json"},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as error:
        raise MunicipalityLocationError(
            "Não foi possível validar a localização do município no IBGE."
        ) from error

    geometry = _extract_geometry(payload, int(ibge_code))
    longitude, latitude = _representative_point(geometry)
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        raise MunicipalityLocationError(
            "O IBGE retornou coordenadas inválidas para o município selecionado."
        )
    return latitude, longitude


def _parse_municipality(item: object) -> BrazilianMunicipality | None:
    if not isinstance(item, dict):
        return None
    state = _extract_state(item)
    if state is None:
        return None

    try:
        ibge_code = int(item["id"])
    except (KeyError, TypeError, ValueError):
        return None
    name = str(item.get("nome") or "").strip()
    state_code = str(state.get("sigla") or "").strip().upper()
    state_name = str(state.get("nome") or "").strip()
    if not name or len(state_code) != 2 or not state_name:
        return None
    return BrazilianMunicipality(ibge_code, name, state_code, state_name)


def _extract_state(item: dict[str, object]) -> dict[str, object] | None:
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
    return None


def _extract_geometry(payload: object, ibge_code: int) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise MunicipalityLocationError(
            "A malha do município retornou um formato inesperado."
        )
    features = payload.get("features")
    if not isinstance(features, list):
        raise MunicipalityLocationError(
            "A malha do município retornou um formato inesperado."
        )
    for feature in features:
        if not isinstance(feature, dict):
            continue
        properties = feature.get("properties")
        geometry = feature.get("geometry")
        code = properties.get("codarea") if isinstance(properties, dict) else None
        if str(code) == str(ibge_code) and isinstance(geometry, dict):
            return geometry
    raise MunicipalityLocationError(
        "O IBGE não retornou a malha do município selecionado."
    )


def _representative_point(geometry: dict[str, object]) -> tuple[float, float]:
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates")
    if geometry_type == "Polygon" and isinstance(coordinates, list):
        polygons = [coordinates]
    elif geometry_type == "MultiPolygon" and isinstance(coordinates, list):
        polygons = coordinates
    else:
        raise MunicipalityLocationError(
            "A malha do município não contém uma geometria compatível."
        )

    valid_polygons = [
        polygon
        for polygon in polygons
        if isinstance(polygon, list)
        and polygon
        and isinstance(polygon[0], list)
        and len(polygon[0]) >= 3
    ]
    if not valid_polygons:
        raise MunicipalityLocationError(
            "A malha do município não contém coordenadas suficientes."
        )
    polygon = max(valid_polygons, key=lambda item: abs(_ring_area(item[0])))
    return _point_inside_polygon(polygon)


def _point_inside_polygon(polygon: list[object]) -> tuple[float, float]:
    exterior = polygon[0]
    centroid = _ring_centroid(exterior)
    if _contains_point(polygon, centroid):
        return centroid

    longitudes = [float(point[0]) for point in exterior]
    latitudes = [float(point[1]) for point in exterior]
    min_lon, max_lon = min(longitudes), max(longitudes)
    min_lat, max_lat = min(latitudes), max(latitudes)
    candidates: list[tuple[float, float]] = []
    for row in range(1, 20):
        latitude = min_lat + (max_lat - min_lat) * row / 20
        for column in range(1, 20):
            longitude = min_lon + (max_lon - min_lon) * column / 20
            candidate = (longitude, latitude)
            if _contains_point(polygon, candidate):
                candidates.append(candidate)
    if candidates:
        return min(
            candidates,
            key=lambda point: math.dist(point, centroid),
        )
    raise MunicipalityLocationError(
        "Não foi possível determinar um ponto interno seguro para o município."
    )


def _contains_point(
    polygon: list[object],
    point: tuple[float, float],
) -> bool:
    if not _point_in_ring(point, polygon[0]):
        return False
    return not any(
        _point_in_ring(point, ring)
        for ring in polygon[1:]
        if isinstance(ring, list) and len(ring) >= 3
    )


def _point_in_ring(point: tuple[float, float], ring: list[object]) -> bool:
    longitude, latitude = point
    inside = False
    previous = ring[-1]
    for current in ring:
        current_lon, current_lat = float(current[0]), float(current[1])
        previous_lon, previous_lat = float(previous[0]), float(previous[1])
        crosses = (current_lat > latitude) != (previous_lat > latitude)
        if crosses:
            intersection = (
                (previous_lon - current_lon)
                * (latitude - current_lat)
                / (previous_lat - current_lat)
                + current_lon
            )
            if longitude < intersection:
                inside = not inside
        previous = current
    return inside


def _ring_area(ring: list[object]) -> float:
    area = 0.0
    previous = ring[-1]
    for current in ring:
        area += float(previous[0]) * float(current[1])
        area -= float(current[0]) * float(previous[1])
        previous = current
    return area / 2


def _ring_centroid(ring: list[object]) -> tuple[float, float]:
    area = _ring_area(ring)
    if math.isclose(area, 0.0):
        points = ring[:-1] if ring[0] == ring[-1] else ring
        return (
            sum(float(point[0]) for point in points) / len(points),
            sum(float(point[1]) for point in points) / len(points),
        )

    longitude_sum = 0.0
    latitude_sum = 0.0
    previous = ring[-1]
    for current in ring:
        cross = (
            float(previous[0]) * float(current[1])
            - float(current[0]) * float(previous[1])
        )
        longitude_sum += (float(previous[0]) + float(current[0])) * cross
        latitude_sum += (float(previous[1]) + float(current[1])) * cross
        previous = current
    factor = 1 / (6 * area)
    return longitude_sum * factor, latitude_sum * factor
