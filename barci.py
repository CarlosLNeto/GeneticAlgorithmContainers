from vpython import *
import random

nome1 = "Sabrina Millane"
nome2 = "Carlos Alves"
nome3 = "David Augusto"
nome4 = "Afonso Garcia"
nome5 = "José Coelho"

# scene = canvas(title=f'\n\n\n\nEngenharia da Computação: Carregamento de containers 2024/01 \n Alunos: {nome1}, {nome2},{nome3}, {nome4}, {nome5}\n\n', width=1600, height=800, center=vector(0, 5, 0))

# imagem1 = 'uea_logo_vertical_verde.png'
# imagem2 = 'images.png'
# # Adicionando os nomes como texto na cena
# texto_nomes = f"{nome1}\n{nome2}\n{nome3}\n{nome4}\n{nome5}"
# text(text=texto_nomes, pos=vector(-35, 10, 0), height=1, depth=0.1, color=color.white)
# uealogo = box(pos=vector(-41, 10, 0), size=vector(10,5, 1), texture = imagem1 )
# estlogo = box(pos=vector(-41, 5, 0), size=vector(10,5, 1), texture = imagem2 )

textura2 = textures.stones
textura1 = textures.wood
textura3 = textures.rough
textura5 = textures.metal
barco = box(
    pos=vector(-6, 0, 0),
    size=vector(19, 3, 7),
    color=vector(0.1, 0.1, 0.4),
    texture=textura5,
)
faixa = box(
    pos=vector(-6, 0, 0), size=vector(19, 0.9, 7.2), color=vector(0.5, 0.1, 0.1)
)
cabine = box(pos=vector(-13, 5, 0), size=vector(3, 7, 3), color=vector(0.8, 0.7, 0.5))
cabineDois = box(
    pos=vector(-12, 9.5, 0), size=vector(3, 2, 3), color=vector(0.8, 0.7, 0.5)
)
chamineDobarco = cylinder(
    pos=vector(-14, 8.5, 0),
    axis=vector(0, 3.5, 0),
    radius=0.5,
    color=vector(0.5, 0.1, 0.1),
    texture=textura5,
)
proa = pyramid(
    pos=vector(3.5, 0, 0),
    size=vector(2, 3, 7.1),
    color=vector(0.1, 0.1, 0.4),
    axis=vector(1, 0, 0),
    texture=textura5,
)
janela = sphere(pos=vector(2, 0.9, 3.2), radius=0.5, color=color.white)
janela1 = sphere(pos=vector(-11, 0.9, 3.2), radius=0.5, color=color.white)
janela2 = sphere(pos=vector(-12, 0.9, 3.2), radius=0.5, color=color.white)
janela3 = sphere(pos=vector(-13, 0.9, 3.2), radius=0.5, color=color.white)
janelacabine1 = box(
    pos=vector(-11.4, 9.5, 0), size=vector(2, 0.9, 2), color=color.white
)
janelacabine2 = box(pos=vector(-13, 6, 1.2), size=vector(1, 1, 1), color=color.white)
chao = box(
    pos=vector(-5, 0, 15),
    size=vector(32, 1, 13),
    color=vector(0.5, 0.5, 0.5),
    texture=textura3,
)
agua = box(
    pos=vector(0, -1, 0), size=vector(43, 1, 20), color=color.cyan, texture=textura3
)
nomeDoBarco = text(
    text="Jaraqui", pos=vector(-6.5, 1, 4), height=1, depth=0.1, color=color.white
)

# Adicionando grades ao redor do barco
altura_grade = 0.5
distancia_grade = 0.1

for x in range(-13, 3, 2):
    cylinder(
        pos=vector(x - 2.5, 1.5, 3.5),
        axis=vector(0, altura_grade, 0),
        radius=0.1,
        color=color.gray(0.5),
    )
    cylinder(
        pos=vector(x - 2.5, 1.5, -3.5),
        axis=vector(0, altura_grade, 0),
        radius=0.1,
        color=color.gray(0.5),
    )

for x in range(-13, 3, 2):
    cylinder(
        pos=vector(x - 2.5, 2, 3.5),
        axis=vector(5, 0, 0),
        radius=0.05,
        color=color.gray(0.5),
    )
    cylinder(
        pos=vector(x - 2.5, 2, -3.5),
        axis=vector(5, 0, 0),
        radius=0.05,
        color=color.gray(0.5),
    )


cores = [
    color.red,
    color.green,
    color.blue,
    color.yellow,
    color.orange,
    color.purple,
    color.cyan,
    color.magenta,
    color.white,
    color.black,
    vector(0.5, 0.1, 0.1),
    vector(1.0, 0.5, 0.5),
    vector(0.5, 0.0, 0.5),
    vector(0.4, 0.2, 0.0),
    vector(0.6, 0.8, 1.0),
    vector(0.25, 0.88, 0.82),
    vector(1.0, 0.55, 0.35),
    vector(0.7, 0.5, 0.8),
    vector(0.7, 0.5, 1.0),
    vector(0.5, 0.5, 0.0),
    vector(0.8, 0.4, 0.6),
    vector(0.2, 0.6, 0.4),
    vector(0.1, 0.3, 0.7),
]


cargas = []
for i in range(2):
    for j in range(5):
        for k in range(3):
            x = -12 + j * 3
            y = 2 if i == 0 else 1
            z = 12.5 + k * 1.5
            cor_aleatoria = random.choice(cores)
            carga = box(
                pos=vector(x, y, z), size=vector(3, 1, 1.3), color=cor_aleatoria
            )
            cargas.append(carga)


base_guindaste = box(
    pos=vector(9, 10, 15), size=vector(2, 20, 2), color=color.orange, texture=textura5
)
braco_guindaste = box(
    pos=vector(9, 20, 15),
    size=vector(10, 0.5, 0.5),
    color=color.yellow,
    axis=vector(1, 0, 0),
    texture=textura5,
)
gancho = box(
    pos=vector(-8, 6, 15), size=vector(1, 1, 1), color=color.red, texture=textura5
)

gancho_original_pos = gancho.pos

carga_posicoes = []

# 12 containers no andar principal
for i in range(2):
    for j in range(3):
        for k in range(4):
            carga_posicoes.append(vector(-8 + k * 3, 2 + i * 1.1, -1 + j * 1.3))

# 6 containers no terceiro andar
posicoes_terceiro_andar = [
    vector(-8 + 1 * 3, 4, -1 + 1 * 1.3),
    vector(-8 + 1 * 3, 4, -1 + 2 * 1.3),
    vector(-8 + 1 * 3, 4, -1 + 0 * 1.3),
    vector(-8 + 2 * 3, 4, -1 + 1 * 1.3),
    vector(-8 + 2 * 3, 4, -1 + 2 * 1.3),
    vector(-8 + 2 * 3, 4, -1 + 0 * 1.3),
]

altura_segura = 15


def move_gancho(gancho, carga, destino):
    while gancho.pos.z < carga.pos.z:
        rate(200)
        gancho.pos.z += 0.1
        braco_guindaste.axis = vector(
            gancho.pos.x - base_guindaste.pos.x, 0, gancho.pos.z - base_guindaste.pos.z
        )
        braco_guindaste.size.x = braco_guindaste.axis.mag

    while gancho.pos.z > carga.pos.z:
        rate(200)
        gancho.pos.z -= 0.1
        braco_guindaste.axis = vector(
            gancho.pos.x - base_guindaste.pos.x, 0, gancho.pos.z - base_guindaste.pos.z
        )
        braco_guindaste.size.x = braco_guindaste.axis.mag

    while gancho.pos.x < carga.pos.x:
        rate(200)
        gancho.pos.x += 0.1
        braco_guindaste.axis = vector(
            gancho.pos.x - base_guindaste.pos.x, 0, gancho.pos.z - base_guindaste.pos.z
        )
        braco_guindaste.size.x = braco_guindaste.axis.mag

    while gancho.pos.x > carga.pos.x:
        rate(200)
        gancho.pos.x -= 0.1
        braco_guindaste.axis = vector(
            gancho.pos.x - base_guindaste.pos.x, 0, gancho.pos.z - base_guindaste.pos.z
        )
        braco_guindaste.size.x = braco_guindaste.axis.mag

    while gancho.pos.y > carga.pos.y + 1:
        rate(200)
        gancho.pos.y -= 0.1

    carga.pos.y = gancho.pos.y

    while gancho.pos.y < altura_segura:
        rate(200)
        gancho.pos.y += 0.1
        carga.pos.y = gancho.pos.y

    while gancho.pos.x < destino.x:
        rate(200)
        gancho.pos.x += 0.1
        braco_guindaste.axis = vector(
            gancho.pos.x - base_guindaste.pos.x, 0, gancho.pos.z - base_guindaste.pos.z
        )
        braco_guindaste.size.x = braco_guindaste.axis.mag
        carga.pos.x = gancho.pos.x

    while gancho.pos.x > destino.x:
        rate(200)
        gancho.pos.x -= 0.1
        braco_guindaste.axis = vector(
            gancho.pos.x - base_guindaste.pos.x, 0, gancho.pos.z - base_guindaste.pos.z
        )
        braco_guindaste.size.x = braco_guindaste.axis.mag
        carga.pos.x = gancho.pos.x

    while gancho.pos.z < destino.z:
        rate(200)
        gancho.pos.z += 0.1
        braco_guindaste.axis = vector(
            gancho.pos.x - base_guindaste.pos.x, 0, gancho.pos.z - base_guindaste.pos.z
        )
        braco_guindaste.size.x = braco_guindaste.axis.mag
        carga.pos.z = gancho.pos.z

    while gancho.pos.z > destino.z:
        rate(200)
        gancho.pos.z -= 0.1
        braco_guindaste.axis = vector(
            gancho.pos.x - base_guindaste.pos.x, 0, gancho.pos.z - base_guindaste.pos.z
        )
        braco_guindaste.size.x = braco_guindaste.axis.mag
        carga.pos.z = gancho.pos.z

    while gancho.pos.y > destino.y + 1:
        rate(200)
        gancho.pos.y -= 0.1
        carga.pos.y = gancho.pos.y

    carga.pos.y = destino.y

    while gancho.pos.y < gancho_original_pos.y:
        rate(200)
        gancho.pos.y += 0.1

    while gancho.pos.y > gancho_original_pos.y:
        rate(200)
        gancho.pos.y -= 0.1


ordem_cargas_chao = [
    8,
    6,
    7,
    14,
    12,
    13,
    5,
    0,
    4,
    2,
    3,
    10,
    11,
    9,
    1,
    20,
    18,
    25,
    26,
    24,
    19,
    29,
    17,
    27,
    21,
    15,
    28,
    23,
    16,
    22,
]
ordem_cargas_barco = [
    5,
    9,
    1,
    6,
    10,
    2,
    4,
    8,
    0,
    7,
    11,
    3,
    17,
    21,
    13,
    18,
    22,
    14,
    16,
    20,
    12,
    19,
    23,
    15,
    28,
    25,
    29,
    26,
    27,
    24,
]

for idx_chao, idx_barco in zip(ordem_cargas_chao[:24], ordem_cargas_barco[:24]):
    move_gancho(gancho, cargas[idx_chao], carga_posicoes[idx_barco])

for idx_chao, pos_terceiro_andar in zip(
    ordem_cargas_chao[24:], posicoes_terceiro_andar
):
    move_gancho(gancho, cargas[idx_chao], pos_terceiro_andar)

# Abrir a cena em tela cheia
scene.fullscreen = True

# Executar a cena
scene.waitfor("click")  # Espera um clique do usuário para fechar a cena
