import math
from typing import Optional

# Origin: Darregueira, Buenos Aires, Argentina
ORIGIN_LAT = -37.6833
ORIGIN_LON = -62.9667
ORIGIN_NAME = "Darregueira, Buenos Aires"

# Comprehensive Argentine city → (lat, lon) lookup
# Covers all provincial capitals + major cities + towns near the origin
CITY_COORDS: dict[str, tuple[float, float]] = {
    # CABA / Gran Buenos Aires
    "buenos aires": (-34.6037, -58.3816),
    "ciudad autónoma de buenos aires": (-34.6037, -58.3816),
    "capital federal": (-34.6037, -58.3816),
    "caba": (-34.6037, -58.3816),
    "palermo": (-34.5853, -58.4337),
    "belgrano": (-34.5620, -58.4570),
    "la plata": (-34.9205, -57.9536),
    "mar del plata": (-38.0023, -57.5575),
    "bahía blanca": (-38.7196, -62.2724),
    "tandil": (-37.3214, -59.1393),
    "olavarría": (-36.8922, -60.3224),
    "azul": (-36.7752, -59.8581),
    "necochea": (-38.5540, -58.7378),
    "tres arroyos": (-38.3756, -60.2756),
    "pigüé": (-37.6035, -62.4072),
    "pigue": (-37.6035, -62.4072),
    "saavedra": (-37.7559, -62.3508),
    "darregueira": (-37.6833, -62.9667),
    "puan": (-37.5389, -62.7664),
    "puán": (-37.5389, -62.7664),
    "guaminí": (-37.0103, -62.4178),
    "guamini": (-37.0103, -62.4178),
    "coronel suárez": (-37.4608, -61.9248),
    "coronel suarez": (-37.4608, -61.9248),
    "coronel pringles": (-37.9942, -61.3567),
    "coronel dorrego": (-38.7222, -61.2847),
    "monte hermoso": (-38.9842, -61.2983),
    "villa iris": (-38.2000, -63.2167),
    "bordenave": (-37.8500, -63.0167),
    "carhué": (-37.1833, -62.7500),
    "carhue": (-37.1833, -62.7500),
    "adolfo alsina": (-37.1833, -62.7500),   # partido de Carhué
    "punta alta": (-38.8833, -62.0833),
    "punta-alta": (-38.8833, -62.0833),
    "nueve de julio": (-35.4500, -60.8833),  # alias of "9 de julio"
    "san nicolas de los arroyos": (-33.3326, -60.2175),
    "san nicolás de los arroyos": (-33.3326, -60.2175),
    "jacinto arauz": (-38.0833, -63.4500),
    "tornquist": (-38.0833, -62.2333),
    "general san martín": (-37.9789, -63.6035),    # La Pampa (not GBA)
    "general san martin": (-37.9789, -63.6035),    # La Pampa (not GBA)
    # GBA-qualified lookups — used when raw_location contains "G.B.A." context
    "general san martín gba": (-34.5731, -58.5336),
    "general san martin gba": (-34.5731, -58.5336),
    # GBA San Martín — must be explicit so "san martin" doesn't partial-match to La Pampa above
    "san martín": (-34.5731, -58.5336),            # Partido General San Martín, GBA
    "san martin": (-34.5731, -58.5336),
    "gral. san martín": (-34.5731, -58.5336),      # abbreviated form used by ML for GBA partido
    "gral. san martin": (-34.5731, -58.5336),
    "cnel. dorrego": (-38.7222, -61.2847),
    # Abbreviated "Coronel" / "General" forms used by Autocosmos and Kavak
    "cnel. suárez": (-37.4608, -61.9248),
    "cnel. suarez": (-37.4608, -61.9248),
    "cnel. pringles": (-37.9942, -61.3567),
    "cnel. rosales": (-38.8167, -62.0833),
    "cnel. brandsen": (-35.1667, -58.2333),
    "cnel. belisle": (-39.1869, -65.9542),
    "gral. madariaga": (-37.0040, -57.1343),
    "gral. pico": (-35.6573, -63.7574),
    "gral. rodriguez": (-34.6056, -58.9639),
    "gral. rodríguez": (-34.6056, -58.9639),
    "gral. alvear": (-36.0167, -60.0167),
    "gral. ramirez": (-32.1333, -59.7500),
    "gral. ramírez": (-32.1333, -59.7500),
    "gral. fernández oro": (-38.9167, -68.0667),
    "gral. fernandez oro": (-38.9167, -68.0667),
    "gral. pedernera": (-35.6833, -65.0833),
    "gral. acha": (-37.3782, -64.6042),
    "gral. villegas": (-35.0326, -63.0148),
    "gral. conesa": (-40.1041, -64.4559),
    "gral. lagos": (-33.1500, -60.5667),
    "gral. manuel j. campos": (-37.4614, -63.5858),
    # Missing cities found in DB (deals with null distance)
    "coronel rosales": (-38.8167, -62.0833),   # partido BA ~148km
    "maracó": (-35.6587, -63.7597),             # Gral. Pico dept, La Pampa ~250km
    "maraco": (-35.6587, -63.7597),
    "mar de ajo": (-36.7167, -56.6833),
    "san antonio de areco": (-34.2500, -59.4667),
    "achiras": (-33.1667, -64.9833),
    "la consulta": (-33.7333, -69.1167),
    "rio gallegos": (-51.6230, -69.2168),
    "obera": (-27.4833, -55.1167),
    "gerli": (-34.6667, -58.3833),
    "colonia aurora": (-27.5167, -54.0167),
    "arroyo leyes": (-31.4167, -60.5833),
    "carrodilla": (-32.9167, -68.8667),
    "marcos paz": (-34.7833, -58.8333),
    "villa nueva": (-32.4333, -63.2500),
    "lomas del mirador": (-34.6667, -58.5667),
    "ciudad evita": (-34.7167, -58.5667),
    "brinckmann": (-31.5000, -62.5833),
    "franck": (-31.5833, -60.9333),
    "laborde": (-33.1667, -62.8500),
    "gonzalez moreno": (-35.7833, -62.9833),
    "salliqueló": (-36.7500, -62.9500),
    "salliquelo": (-36.7500, -62.9500),
    "pellegrini": (-36.2833, -63.1500),
    "bernasconi": (-37.9000, -63.7333),
    "santa trinidad": (-38.2500, -63.3833),
    "mayor buratovich": (-39.2500, -62.6167),
    "pedro luro": (-39.5167, -62.7000),
    "médanos": (-38.8333, -62.6833),
    "san cayetano": (-38.3440, -59.6112),
    "lobería": (-38.1623, -58.7886),
    "rauch": (-36.7731, -59.0826),
    "benito juárez": (-37.6707, -59.8021),
    "laprida": (-37.5374, -60.7963),
    "bolívar": (-36.2347, -61.1166),
    "pehuajó": (-35.8152, -61.8958),
    "trenque lauquen": (-35.9714, -62.7278),
    "lincoln": (-34.8648, -61.5279),
    "junín": (-34.5909, -60.9456),
    "9 de julio": (-35.4500, -60.8833),
    "bragado": (-35.1167, -60.4833),
    "chivilcoy": (-34.8992, -60.0213),
    "mercedes": (-34.6500, -59.4333),
    "luján": (-34.5698, -59.1065),
    "campana": (-34.1623, -58.9573),
    "zárate": (-34.0986, -59.0272),
    "san nicolás": (-33.3326, -60.2175),
    "pergamino": (-33.8899, -60.5734),
    "rojas": (-34.1954, -60.7348),
    "salto": (-34.2934, -60.2538),
    "quilmes": (-34.7214, -58.2567),
    "lomas de zamora": (-34.7604, -58.4049),
    "lanús": (-34.7072, -58.3914),
    "avellaneda": (-34.6645, -58.3656),
    "morón": (-34.6505, -58.6197),
    "san isidro": (-34.4722, -58.5255),
    "tigre": (-34.4261, -58.5793),
    "san martín": (-34.5748, -58.5413),
    "merlo": (-34.6709, -58.7280),
    "moreno": (-34.6395, -58.7880),
    "florencio varela": (-34.8082, -58.2756),
    "quilmes": (-34.7214, -58.2567),
    # Provincia de Buenos Aires - Resto
    "pinamar": (-37.1122, -56.8699),
    "villa gesell": (-37.2644, -56.9766),
    "miramar": (-38.2717, -57.8411),
    "balcarce": (-37.8399, -58.2562),
    "ayacucho": (-37.1540, -58.4869),
    "dolores": (-36.3129, -57.6782),
    "maipú": (-36.8653, -57.8818),
    "general madariaga": (-37.0040, -57.1343),
    "san clemente del tuyú": (-36.3622, -56.7218),
    "pilar": (-34.4587, -58.9138),
    "zárate": (-34.0986, -59.0272),
    "escobar": (-34.3492, -58.7971),
    "tres de febrero": (-34.6057, -58.5613),
    "ituzaingó": (-34.6567, -58.6686),
    "hurlingham": (-34.5894, -58.6381),
    "esteban echeverría": (-34.8128, -58.4607),
    "almirante brown": (-34.8041, -58.3956),
    "presidente perón": (-34.9190, -58.4410),
    "cañuelas": (-35.0517, -58.7568),
    "general las heras": (-34.9268, -58.9490),
    # Córdoba
    "córdoba": (-31.4201, -64.1888),
    "villa carlos paz": (-31.4235, -64.4970),
    "río cuarto": (-33.1307, -64.3499),
    "san francisco": (-31.4292, -62.0818),
    "villa maría": (-32.4080, -63.2442),
    "alta gracia": (-31.6556, -64.4300),
    "jesús maría": (-30.9824, -64.0959),
    "villa del totoral": (-30.9333, -63.9833),
    "cosquín": (-31.2428, -64.4722),
    "la falda": (-31.0869, -64.4884),
    "bell ville": (-32.6269, -62.6869),
    "río tercero": (-32.1741, -64.1095),
    "marcos juárez": (-32.6980, -62.1056),
    # Santa Fe
    "santa fe": (-31.6333, -60.7000),
    "rosario": (-32.9468, -60.6393),
    "rafaela": (-31.2517, -61.4869),
    "venado tuerto": (-33.7461, -61.9690),
    "reconquista": (-29.1458, -59.6439),
    "resistencia": (-27.4514, -58.9868),  # actually Chaco but often confused
    "villa constitución": (-33.2310, -60.3378),
    "casilda": (-33.0447, -61.1681),
    # Mendoza
    "mendoza": (-32.8908, -68.8272),
    "san rafael": (-34.6177, -68.3301),
    "godoy cruz": (-32.9239, -68.8459),
    "luján de cuyo": (-33.0439, -68.8781),
    "maipú": (-32.9820, -68.7900),
    "rivadavia": (-33.1830, -68.4610),
    # San Juan
    "san juan": (-31.5375, -68.5364),
    "rivadavia": (-31.5347, -68.5782),
    "rawson": (-31.5349, -68.5237),
    # La Rioja
    "la rioja": (-29.4132, -66.8560),
    "chilecito": (-29.1647, -67.4985),
    # Catamarca
    "san fernando del valle de catamarca": (-28.4696, -65.7795),
    "catamarca": (-28.4696, -65.7795),
    "san fernando": (-28.4696, -65.7795),
    # Salta
    "salta": (-24.7859, -65.4117),
    "tartagal": (-22.5223, -63.8015),
    "orán": (-23.1334, -64.3197),
    "metán": (-25.4982, -64.9731),
    # Jujuy
    "san salvador de jujuy": (-24.1858, -65.2995),
    "jujuy": (-24.1858, -65.2995),
    "palpalá": (-24.2574, -65.2071),
    "libertador general san martín": (-23.8096, -64.7934),
    # Tucumán
    "san miguel de tucumán": (-26.8083, -65.2176),
    "tucumán": (-26.8083, -65.2176),
    "tafí viejo": (-26.7289, -65.2647),
    "yerba buena": (-26.8157, -65.3236),
    "banda del río salí": (-26.8399, -65.1660),
    # Santiago del Estero
    "santiago del estero": (-27.7951, -64.2615),
    "la banda": (-27.7346, -64.2408),
    "termas de río hondo": (-27.4924, -64.8594),
    # Chaco
    "resistencia": (-27.4514, -58.9868),
    "barranqueras": (-27.4875, -58.9403),
    "sáenz peña": (-26.7895, -60.4503),
    "villa ángela": (-27.5739, -60.7144),
    # Formosa
    "formosa": (-26.1775, -58.1781),
    "clorinda": (-25.2830, -57.7241),
    # Misiones
    "posadas": (-27.3671, -55.8960),
    "oberá": (-27.4869, -55.1192),
    "eldorado": (-26.4028, -54.6340),
    "iguazú": (-25.5972, -54.5714),
    "puerto iguazú": (-25.5972, -54.5714),
    # Corrientes
    "corrientes": (-27.4806, -58.8341),
    "goya": (-29.1415, -59.2673),
    "mercedes": (-29.1807, -58.0789),
    "curuzú cuatiá": (-29.7943, -57.9985),
    # Entre Ríos
    "paraná": (-31.7333, -60.5333),
    "concordia": (-31.3924, -58.0211),
    "gualeguaychú": (-33.0095, -58.5234),
    "concepción del uruguay": (-32.4853, -58.2373),
    "colón": (-32.2236, -58.1478),
    # Neuquén
    "neuquén": (-38.9516, -68.0591),
    "san martín de los andes": (-40.1573, -71.3550),
    "villa la angostura": (-40.7632, -71.6468),
    "zapala": (-38.8990, -70.0670),
    "plottier": (-38.9601, -68.2308),
    # Río Negro
    "viedma": (-40.8135, -62.9967),
    "san carlos de bariloche": (-41.1335, -71.3103),
    "bariloche": (-41.1335, -71.3103),
    "general roca": (-39.0355, -67.5845),
    "cipolletti": (-38.9411, -67.9922),
    "villa regina": (-39.0975, -67.0768),
    "allen": (-38.9795, -67.8280),
    # Chubut
    "rawson": (-43.3002, -65.1023),
    "comodoro rivadavia": (-45.8644, -67.4953),
    "trelew": (-43.2489, -65.3026),
    "puerto madryn": (-42.7692, -65.0385),
    "esquel": (-42.9108, -71.3178),
    # Santa Cruz
    "río gallegos": (-51.6230, -69.2168),
    "caleta olivia": (-46.4380, -67.5216),
    "pico truncado": (-46.7940, -67.9680),
    "puerto deseado": (-47.7500, -65.9000),
    # Tierra del Fuego
    "ushuaia": (-54.8019, -68.3030),
    "río grande": (-53.7878, -67.7081),
    # Córdoba — must be before "santa rosa" to win the longest-match tie-break
    "santa rosa de calamuchita": (-32.0667, -64.5500),
    "villa general belgrano": (-31.9833, -64.5667),
    # La Pampa
    "santa rosa": (-36.6167, -64.2833),
    "general pico": (-35.6573, -63.7574),
    "toay": (-36.6701, -64.3756),
    "eduardo castex": (-35.9155, -64.2927),
    # San Luis
    "san luis": (-33.2950, -66.3356),
    "villa mercedes": (-33.6742, -65.4598),
    "merlo": (-32.3459, -65.0169),
    # Gran Buenos Aires — partidos / localidades
    "vicente lópez": (-34.5250, -58.4750),
    "vicente lopez": (-34.5250, -58.4750),
    "ramos mejia": (-34.6444, -58.5625),
    "ramos mejía": (-34.6444, -58.5625),
    "san justo": (-34.6766, -58.5550),
    "berazategui": (-34.7617, -58.2108),
    "berazategui oeste": (-34.7617, -58.2108),
    "castelar": (-34.6533, -58.6408),
    "ituzaingo": (-34.6561, -58.6714),
    "ituzaingó": (-34.6561, -58.6714),
    "banfield": (-34.7467, -58.4000),
    "malvinas argentinas": (-34.4167, -58.7667),
    "bernal": (-34.7083, -58.2753),
    "adrogue": (-34.7975, -58.3897),
    "adrogué": (-34.7975, -58.3897),
    "villa adelina": (-34.5256, -58.5528),
    "ezeiza": (-34.8500, -58.5167),
    "ingeniero maschwitz": (-34.3806, -58.7628),
    "martinez": (-34.4875, -58.4958),
    "martínez": (-34.4875, -58.4958),
    "temperley": (-34.7800, -58.4028),
    "moron": (-34.6525, -58.6197),
    "grand bourg": (-34.4667, -58.7500),
    "el palomar": (-34.6100, -58.5931),
    "lanus": (-34.7069, -58.3931),
    "claypole": (-34.8186, -58.3431),
    "monte grande": (-34.8272, -58.4597),
    "remedios de escalada": (-34.7275, -58.3956),
    "guillermo hudson": (-34.7833, -58.1667),
    "la tablada": (-34.6806, -58.5644),
    "beccar": (-34.4581, -58.5378),
    "del viso": (-34.4583, -58.7833),
    "san antonio de padua": (-34.6644, -58.7344),
    "martin coronado": (-34.5917, -58.5833),
    "boulogne sur mer": (-34.4917, -58.5583),
    "florida oeste": (-34.5406, -58.5178),
    "villa martelli": (-34.5611, -58.5006),
    "garin": (-34.4208, -58.7347),
    "garín": (-34.4208, -58.7347),
    "la matanza": (-34.7692, -58.6133),
    "caseros": (-34.6064, -58.5631),
    "general rodriguez": (-34.6056, -58.9639),
    "ciudad jardín lomas del palomar": (-34.6056, -58.5944),
    "san cristobal": (-34.6200, -58.5400),
    # Buenos Aires interior
    "carlos casares": (-35.6197, -61.3644),
    "coronel brandsen": (-35.1667, -58.2333),
    "tapalqué": (-36.3564, -60.0192),
    "tapalgué": (-36.3564, -60.0192),
    "san vicente": (-35.0244, -58.4253),
    "ensenada": (-34.8594, -57.9253),
    "chabás": (-33.2536, -61.3617),
    "general alvear": (-36.0167, -60.0167),
    "la picada": (-31.6833, -60.2167),
    "arroyito": (-31.4167, -63.0500),
    # Córdoba — extra localidades
    "malagueño": (-31.4844, -64.2781),
    "oncativo": (-31.9178, -63.6800),
    "rio segundo": (-31.6536, -63.9069),
    "río segundo": (-31.6536, -63.9069),
    "unquillo": (-31.2333, -64.3167),
    "mendiolaza": (-31.2333, -64.2833),
    "la calera": (-31.3167, -64.3333),
    "villa ciudad parque los reartes": (-31.9167, -64.6500),
    "calamuchita": (-32.0500, -64.5333),
    # Santa Fe — extra localidades
    "santo tome": (-31.6667, -60.7667),
    "santo tomé": (-31.6667, -60.7667),
    "galvez": (-32.0333, -61.2167),
    "gálvez": (-32.0333, -61.2167),
    "hersilia": (-29.9333, -61.6500),
    "arroyo seco": (-33.1500, -60.5000),
    "esperanza": (-31.4486, -60.9328),
    "cañada de gomez": (-32.8200, -61.3900),
    "capitán bermúdez": (-32.8269, -60.7142),
    "capitan bermudez": (-32.8269, -60.7142),
    "recreo": (-31.4833, -60.7167),
    # Mendoza — extra
    "guaymallén": (-32.8892, -68.7697),
    "guaymallen": (-32.8892, -68.7697),
    "rodeo de la cruz": (-33.1167, -68.7667),
    "rodeo del medio": (-33.1000, -68.7000),
    "la puntilla": (-32.8800, -68.9900),
    "gobernador benegas": (-32.9167, -68.8500),
    "chacras de coria": (-33.0167, -68.8500),
    "maipu": (-32.9822, -68.7728),
    # Entre Ríos — extra
    "aldea valle maria": (-32.3167, -60.6667),
    "aldea valle maría": (-32.3167, -60.6667),
    "lujan de cuyo": (-33.0439, -68.8781),
    "parana": (-31.7333, -60.5333),
    "las colonias": (-31.2500, -61.1333),
    "general ramirez": (-32.1333, -59.7500),
    "general ramírez": (-32.1333, -59.7500),
    # Neuquén — extra
    "centenario": (-38.8333, -68.1333),
    "general fernández oro": (-38.9167, -68.0667),
    "general fernandez oro": (-38.9167, -68.0667),
    # San Luis — extra
    "general pedernera": (-35.6833, -65.0833),
    "villa general mitre": (-35.6833, -65.0833),
    # Chaco — extra
    "machagai": (-26.9333, -60.0500),
    "presidencia roque saenz pena": (-26.7833, -60.4333),
    "presidencia roque sáenz peña": (-26.7833, -60.4333),
    # Salta — extra
    "joaquín v. gonzález": (-25.1000, -64.1333),
    "joaquin v. gonzalez": (-25.1000, -64.1333),
    # Corrientes — extra
    "monte caseros": (-30.2583, -57.6403),
    "saladas": (-28.2500, -58.6333),
    "gobernador igr.valentin virasoro": (-28.0500, -56.0167),
    # Tucumán — extra
    "tafi viejo": (-26.7289, -65.2647),
    # Chubut — extra
    "escalante": (-45.8644, -67.4953),
    # Misc
    "barrio porvenir": (-32.9500, -68.8000),
    "villa maria": (-32.4080, -63.2442),
    "santo domingo": (-30.7167, -60.6000),
    "san cristóbal": (-30.3167, -61.2333),
    "juan bautista alberdi": (-27.5833, -65.6167),
    "victoria": (-32.6167, -60.1500),
    # ─── La Pampa — localidades (GeoNames, pop ≥ 500, within/near 400km) ───
    "guatraché": (-37.6707, -63.5358),
    "guatrache": (-37.6707, -63.5358),
    "general manuel j. campos": (-37.4614, -63.5858),
    "general san martín": (-37.9789, -63.6035),
    "bernasconi": (-37.9023, -63.7430),
    "alpachiri": (-37.3768, -63.7732),
    "macachín": (-37.1370, -63.6667),
    "macachin": (-37.1370, -63.6667),
    "doblas": (-37.1505, -64.0115),
    "miguel riglos": (-36.8537, -63.6886),
    "ataliva roca": (-37.0315, -64.2843),
    "lonquimay": (-36.4664, -63.6236),
    "catriló": (-36.4067, -63.4233),
    "catrilo": (-36.4067, -63.4233),
    "general acha": (-37.3782, -64.6042),
    "uriburu": (-36.5070, -63.8621),
    "anguil": (-36.5258, -64.0101),
    "la adela": (-38.9833, -64.0833),
    "río colorado": (-38.9908, -64.0957),
    "rio colorado": (-38.9908, -64.0957),
    "colonia barón": (-36.1536, -63.8552),
    "colonia baron": (-36.1536, -63.8552),
    "quemú quemú": (-36.0548, -63.5659),
    "quemu quemu": (-36.0548, -63.5659),
    "winifreda": (-36.2264, -64.2339),
    "trenel": (-35.6983, -64.1333),
    "arata": (-35.6390, -64.3562),
    "caleufú": (-35.5945, -64.5587),
    "caleufu": (-35.5945, -64.5587),
    "victorica": (-36.2165, -65.4371),
    "telén": (-36.2640, -65.5104),
    "telen": (-36.2640, -65.5104),
    "intendente alvear": (-35.2347, -63.5921),
    "alta italia": (-35.3344, -64.1171),
    "embajador martini": (-35.3873, -64.2828),
    "la maruja": (-35.6738, -64.9402),
    "ingeniero luiggi": (-35.3885, -64.4674),
    "parera": (-35.1468, -64.5023),
    "realicó": (-35.0382, -64.2463),
    "realico": (-35.0382, -64.2463),
    "rancul": (-35.0678, -64.6842),
    "italó": (-34.7920, -63.7813),
    "italo": (-34.7920, -63.7813),
    "hipólito bouchard": (-34.7239, -63.5073),
    "hipolito bouchard": (-34.7239, -63.5073),
    "huinca renancó": (-34.8413, -64.3746),
    "huinca renanco": (-34.8413, -64.3746),
    "villa huidobro": (-34.8383, -64.5869),
    "serrano": (-34.4716, -63.5382),
    "jovita": (-34.5189, -63.9440),
    "mattaldi": (-34.4820, -64.1719),
    "buena esperanza": (-34.7579, -65.2536),
    "unión": (-35.1538, -65.9462),
    "union": (-35.1538, -65.9462),
    "santa isabel": (-36.2302, -66.9390),
    "del campillo": (-34.3759, -64.4965),
    "laboulaye": (-34.1269, -63.3908),
    # ─── Buenos Aires province — towns within 400km missing ───
    "punta alta": (-38.8805, -62.0750),
    "tres algarrobos": (-35.1956, -62.7742),
    "alfredo demarchi": (-35.2924, -61.4057),
    "nueve de julio": (-35.4439, -60.8846),
    "general villegas": (-35.0326, -63.0148),
    "veinticinco de mayo": (-35.4325, -60.1716),
    "25 de mayo": (-35.4325, -60.1716),
    "saladillo": (-35.6388, -59.7794),
    "ayacucho": (-37.1528, -58.4884),
    # ─── Santa Fe — extra ───
    "rufino": (-34.2624, -62.7113),
    "bernardo larroudé": (-35.0265, -63.5820),
    "bernardo larroude": (-35.0265, -63.5820),
    # ─── Río Negro — towns within 400km missing ───
    "choele choel": (-39.2894, -65.6606),
    "darwin": (-39.2033, -65.7395),
    "general conesa": (-40.1041, -64.4559),
    "fray luis beltrán": (-39.3137, -65.7600),
    "fray luis beltran": (-39.3137, -65.7600),
    "lamarque": (-39.4252, -65.7022),
    "coronel belisle": (-39.1869, -65.9542),
    "chimpay": (-39.1654, -66.1464),
    "carmen de patagones": (-40.8025, -62.9835),
    "san antonio oeste": (-40.7319, -64.9477),
    "chichinales": (-39.1148, -66.9423),
    "general enrique godoy": (-39.0789, -67.1581),
    # ─── Major GBA/national cities missing from top-population list ───
    "josé c. paz": (-34.5154, -58.7681),
    "jose c. paz": (-34.5154, -58.7681),
    "san miguel": (-34.5433, -58.7123),
    "san nicolás de los arroyos": (-33.3342, -60.2108),
    "san nicolas de los arroyos": (-33.3342, -60.2108),
    "gobernador gálvez": (-33.0302, -60.6405),
    "gobernador galvez": (-33.0302, -60.6405),
    # Gran Buenos Aires — barrios/localidades faltantes
    "haedo": (-34.6444, -58.5928),
    "villa ballester": (-34.5281, -58.5528),
    "city bell": (-34.8833, -58.0500),
    "villa bosch": (-34.5972, -58.6483),
    "villa luzuriaga": (-34.6806, -58.6167),
    "isidro casanova": (-34.7000, -58.6500),
    "villa raffo": (-34.5917, -58.5417),
    "canning": (-34.8833, -58.5167),
    "llavallol": (-34.7667, -58.4167),
    "glew": (-34.8833, -58.3833),
    "ringuelet": (-34.9167, -57.9667),
    "munro": (-34.5333, -58.5167),
    "san andres": (-34.4833, -58.5333),
    "gonzalez catán": (-34.7833, -58.7167),
    "gonzalez catan": (-34.7833, -58.7167),
    # Santa Fe — barrios/localidades
    "general lagos": (-33.1500, -60.5667),
    "funes": (-32.9167, -60.8167),
    "angelica": (-31.5667, -62.0667),
    "angélica": (-31.5667, -62.0667),
    "plaza clucellas": (-31.6167, -62.0833),
    "freyre": (-31.1667, -62.2167),
    # Córdoba — barrios/localidades
    "rio ceballos": (-31.1667, -64.3167),
    "río ceballos": (-31.1667, -64.3167),
    "villa giardino": (-31.0667, -64.4833),
    "villa manzano": (-31.1167, -64.5167),
    "sampacho": (-33.3833, -64.7167),
    "villa nueva": (-32.4333, -63.2500),
    # Neuquén — extra
    "plaza huincul": (-38.9333, -69.2000),
    # La Pampa — extra
    "grunbein": (-36.5833, -63.9667),
    "grünbein": (-36.5833, -63.9667),
    # Chaco — far
    "tres isletas": (-26.3367, -60.4253),
    # Entre Ríos — extra
    "crespo": (-32.0333, -60.3167),
    # GBA — localidades faltantes (detectadas via listings con distancia incorrecta)
    "tortuguitas": (-34.4667, -58.7500),         # Partido Malvinas Argentinas, GBA
    "grand bourg": (-34.4833, -58.7167),          # Partido Malvinas Argentinas, GBA
    "los polvorines": (-34.5000, -58.7000),       # Partido Malvinas Argentinas, GBA
    "ing. pablo nogués": (-34.4833, -58.7833),    # Partido Malvinas Argentinas, GBA
    "pablo nogues": (-34.4833, -58.7833),
    "pablo nogués": (-34.4833, -58.7833),
    "joe pablo nogues": (-34.4833, -58.7833),
    "el talar": (-34.4167, -58.6833),             # Partido Tigre, GBA
    "rincon de milberg": (-34.3833, -58.6667),    # Partido Tigre, GBA
    "rincón de milberg": (-34.3833, -58.6667),
    "benavidez": (-34.4000, -58.6833),            # Partido Tigre, GBA
    "benavídez": (-34.4000, -58.6833),
    "general pacheco": (-34.4500, -58.6500),      # Partido Tigre, GBA
    "dique luján": (-34.3667, -58.7500),          # Partido Tigre, GBA
    "dique lujan": (-34.3667, -58.7500),
    "open door": (-34.3833, -58.8833),            # Partido Luján, GBA
    "maquinista savio": (-34.3833, -58.6833),     # Partido Escobar
    "la lonja": (-34.3667, -58.8167),             # Partido Pilar
    "del viso": (-34.4333, -58.8500),             # Partido Pilar
    "presidente derqui": (-34.4833, -58.8667),    # Partido Pilar
    "villa rosa": (-34.4667, -58.9000),           # Partido Pilar
    "fátima": (-34.4167, -59.0167),               # Partido Pilar
    "fatima": (-34.4167, -59.0167),
    # Mendoza — localidades faltantes
    "tupungato": (-33.3667, -69.1333),            # Valle de Uco, Mendoza
    "tunuyán": (-33.5667, -69.0167),              # Valle de Uco, Mendoza
    "tunuyan": (-33.5667, -69.0167),
    "san carlos": (-33.7667, -69.0500),           # Valle de Uco, Mendoza
    "rivadavia": (-33.1833, -68.4667),            # Mendoza (not to confuse with Rivadavia BA)
    "junín mendoza": (-33.1333, -68.4667),
    "lavalle": (-32.7167, -68.5667),              # Mendoza
    "las heras": (-32.8500, -68.8167),            # Mendoza (not GBA)
    "luján de cuyo": (-33.0500, -68.8833),        # Mendoza
    "lujan de cuyo": (-33.0500, -68.8833),
    "maipú mendoza": (-32.9833, -68.7833),
    "godoy cruz": (-32.9167, -68.8333),           # Gran Mendoza
    "guaymallén": (-32.8833, -68.7833),           # Gran Mendoza
    "guaymallen": (-32.8833, -68.7833),
    # Buenos Aires provincia — localidades faltantes
    "san pedro": (-33.6833, -59.6667),            # Partido San Pedro, BA
    "capitan sarmiento": (-34.1833, -59.7833),    # BA provincia
    "capitán sarmiento": (-34.1833, -59.7833),
    "navarro": (-35.0000, -59.2833),              # BA provincia
    "roque perez": (-35.4000, -59.3333),          # BA provincia
    "roque pérez": (-35.4000, -59.3333),
    "saladillo": (-35.6333, -59.7667),            # BA provincia
    "monte": (-35.4333, -58.8000),                # Partido Monte, BA
    "chascomús": (-35.5667, -58.0000),            # BA provincia
    "chascomus": (-35.5667, -58.0000),
    "brandsen": (-35.1667, -58.2333),             # BA provincia
    "lobos": (-35.1833, -59.0833),                # BA provincia
    "gral. belgrano": (-35.7667, -58.5000),       # BA provincia
    "general belgrano": (-35.7667, -58.5000),
    "punta indio": (-35.2667, -57.2333),          # BA provincia
}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance in km between two lat/lon points."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def distance_from_darregueira(lat: Optional[float], lon: Optional[float]) -> Optional[float]:
    """Return km distance from Darregueira, or None if coords unavailable."""
    if lat is None or lon is None:
        return None
    return haversine_km(ORIGIN_LAT, ORIGIN_LON, lat, lon)


def city_to_coords(city_name: str) -> Optional[tuple[float, float]]:
    """Look up (lat, lon) for an Argentine city name. Case-insensitive."""
    if not city_name:
        return None
    key = city_name.lower().strip()
    # Direct match
    if key in CITY_COORDS:
        return CITY_COORDS[key]
    # Partial match — city_name might be "Bahía Blanca, Buenos Aires".
    # Require at least 4 chars to avoid accidental short-string matches.
    # Prefer the LONGEST matching key to avoid "santa rosa" swallowing
    # "santa rosa de calamuchita" or similar ambiguous substring collisions.
    best_known, best_coords = None, None
    for known, coords in CITY_COORDS.items():
        if len(known) < 4:
            continue
        if known in key or key in known:
            if best_known is None or len(known) > len(best_known):
                best_known, best_coords = known, coords
    return best_coords


def coords_from_listing_dict(listing: dict) -> tuple[Optional[float], Optional[float], Optional[str]]:
    """
    Extract (lat, lon, city_name) from a listing dict.
    Tries raw_data first (ML seller_address), then explicit lat/lon fields,
    then city name lookup.
    """
    raw = listing.get("raw_data") or {}

    # MercadoLibre: seller_address has lat/lon
    seller_addr = raw.get("seller_address") or {}
    lat = seller_addr.get("latitude")
    lon = seller_addr.get("longitude")
    city_name = None
    city_obj = seller_addr.get("city") or {}
    if isinstance(city_obj, dict):
        city_name = city_obj.get("name")
    elif isinstance(city_obj, str):
        city_name = city_obj
    state_obj = seller_addr.get("state") or {}
    if isinstance(state_obj, dict):
        state_name = state_obj.get("name", "")
    else:
        state_name = ""

    if lat is not None and lon is not None:
        try:
            return float(lat), float(lon), city_name or state_name
        except (ValueError, TypeError):
            pass

    # Generic / HTML scrapers: try explicit fields
    lat = listing.get("seller_lat") or raw.get("latitude") or raw.get("lat")
    lon = listing.get("seller_lon") or raw.get("longitude") or raw.get("lng") or raw.get("lon")

    # Build city name: check listing dict first, then raw_data fields
    # NOTE: wrap the conditional at the end in parens to avoid precedence bug
    raw_location = raw.get("location") or ""
    # "Capital Federal - Capital Federal" → "Capital Federal"
    raw_city = raw_location.split(" - ")[0].strip() if raw_location else ""

    city_name = (
        listing.get("seller_city")
        or raw.get("city")
        or raw.get("cityName")
        or raw_city
        or raw.get("branchCity")
        or (raw.get("branch", {}).get("city") if isinstance(raw.get("branch"), dict) else None)
    )

    if lat is not None and lon is not None:
        try:
            return float(lat), float(lon), city_name
        except (ValueError, TypeError):
            pass

    # Fall back to city name → coords lookup.
    # When the full location string contains a GBA/province indicator, try the
    # province-qualified key first so ambiguous city names (e.g. "General San Martín"
    # which exists in both GBA and La Pampa) resolve to the correct province.
    if city_name:
        loc_lower = raw_location.lower() if raw_location else ""
        if "g.b.a." in loc_lower or "gran buenos aires" in loc_lower:
            gba_key = str(city_name).lower().strip() + " gba"
            gba_coords = CITY_COORDS.get(gba_key)
            if gba_coords:
                return gba_coords[0], gba_coords[1], city_name
        coords = city_to_coords(str(city_name))
        if coords:
            return coords[0], coords[1], city_name

    return None, None, city_name
