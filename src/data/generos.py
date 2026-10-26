"""Agrupación de las etiquetas de género de DadaGP en macrogéneros.

DadaGP asigna a cada canción las etiquetas de género de su artista (estilo
Spotify: "classic_rock", "pop_punk", "beatlesque"...), con varias por canción y
"unknown_genre" cuando no hay información. Para analizar los resultados por
género se agrupan en pocos macrogéneros:

1. Cada etiqueta se asigna al primer macrogénero de REGLAS cuya palabra clave
   contiene (el orden resuelve etiquetas mixtas: "pop_punk" es punk,
   "neo_classical_metal" es metal).
2. Cada canción recibe un único macrogénero: el más votado por sus etiquetas;
   los empates se resuelven a favor del más específico (orden de PRIORIDAD).
"""

from collections import Counter
from typing import Iterable

SIN_GENERO = "Sin género"
OTROS = "Otros"

# (macrogénero, palabras clave). Se evalúan en orden; gana la primera coincidencia.
REGLAS = (
    ("Videojuegos y BSO", ("vgm", "video_game", "scorecore", "anime", "final_fantasy",
                           "chiptune", "soundtrack", "wrestling", "library_music")),
    ("Metal", ("metal", "nwobhm", "thrash", "latincore")),
    ("Punk y emo", ("punk", "emo", "dreamo", "hardcore")),
    ("Pop", ("baroque_pop", "chamber_pop", "new_romantic", "neo_mellow")),
    ("Clásica", ("classical", "baroque", "romantic", "renaissance", "early_music",
                 "impressionism", "neoclassicism", "musica_antigua", "guitarra_clasica")),
    ("Jazz y bossa", ("jazz", "bop", "big_band", "bossa", "mpb", "samba", "tropicalia")),
    ("Blues y country", ("blues", "country", "nashville", "rockabilly", "sertanejo")),
    ("Folk y cantautor", ("folk", "singer-songwriter", "cantautor", "chanson", "trova",
                          "nueva_cancion", "celtic", "irish", "fingerstyle",
                          "american_primitive", "chill_guitar", "andina", "polca",
                          "flamenco", "rumba", "lilith")),
    ("Pop", ("pop", "britpop", "madchester", "new_wave", "new_romantic", "mellow_gold",
             "ye_ye", "adult_standards", "soul", "r&b", "urban_contemporary")),
    ("Rock", ("rock", "beatlesque", "british_invasion", "merseybeat", "permanent_wave",
              "grunge", "magyar_alternative", "latin_alternative", "indie", "jam_band",
              "surf_music")),
)

# Desempate: de más específico a más genérico.
PRIORIDAD = ("Videojuegos y BSO", "Clásica", "Jazz y bossa", "Blues y country",
             "Folk y cantautor", "Punk y emo", "Metal", "Pop", "Rock", OTROS)


def macro_etiqueta(etiqueta: str) -> str:
    """Macrogénero de una etiqueta de DadaGP (con o sin el prefijo "genre:")."""
    e = etiqueta.removeprefix("genre:")
    if e == "unknown_genre":
        return SIN_GENERO
    for macro, claves in REGLAS:
        if any(c in e for c in claves):
            return macro
    return OTROS


def macro_cancion(etiquetas: Iterable[str]) -> str:
    """Macrogénero de una canción: voto mayoritario de sus etiquetas conocidas."""
    votos = Counter(m for m in map(macro_etiqueta, etiquetas) if m != SIN_GENERO)
    if not votos:
        return SIN_GENERO
    return max(votos, key=lambda m: (votos[m], -PRIORIDAD.index(m)))
