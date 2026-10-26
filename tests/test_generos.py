from src.data.generos import OTROS, SIN_GENERO, macro_cancion, macro_etiqueta


def test_etiquetas_simples():
    assert macro_etiqueta("genre:classic_rock") == "Rock"
    assert macro_etiqueta("genre:speed_metal") == "Metal"
    assert macro_etiqueta("genre:bossa_nova") == "Jazz y bossa"
    assert macro_etiqueta("genre:singer-songwriter") == "Folk y cantautor"
    assert macro_etiqueta("japanese_vgm") == "Videojuegos y BSO"


def test_etiquetas_mixtas_siguen_el_orden_de_las_reglas():
    assert macro_etiqueta("genre:pop_punk") == "Punk y emo"
    assert macro_etiqueta("genre:neo_classical_metal") == "Metal"
    assert macro_etiqueta("genre:baroque_pop") == "Pop"
    assert macro_etiqueta("genre:new_romantic") == "Pop"
    assert macro_etiqueta("genre:blues_rock") == "Blues y country"


def test_sin_genero_y_otros():
    assert macro_etiqueta("genre:unknown_genre") == SIN_GENERO
    assert macro_etiqueta("genre:reggae") == OTROS
    assert macro_cancion(["genre:unknown_genre"]) == SIN_GENERO
    assert macro_cancion([]) == SIN_GENERO


def test_cancion_por_mayoria():
    tags = ["genre:rock", "genre:classic_rock", "genre:blues_rock"]
    assert macro_cancion(tags) == "Rock"


def test_empate_favorece_al_mas_especifico():
    assert macro_cancion(["genre:rock", "genre:folk"]) == "Folk y cantautor"
    assert macro_cancion(["genre:metal", "genre:classical"]) == "Clásica"
