import cv2
import numpy as np


COULEUR_POINT_KNEEKG = (255, 255, 0)       # jaune (RGB)
COULEUR_CONNEXION_KNEEKG = (255, 140, 50)  # orange (RGB)
COULEUR_FOND = (255, 255, 255)             # blanc (RGB)
RAYON_POINT = 3
EPAISSEUR_CONNEXION = 2


def matrice_translation(tx, ty):
    """Matrice homogene 3x3 de translation."""
    return np.array([
        [1.0, 0.0, float(tx)],
        [0.0, 1.0, float(ty)],
        [0.0, 0.0, 1.0],
    ], dtype=float)


def matrice_scale(sx, sy):
    """Matrice homogene 3x3 de changement d'echelle (agrandissement / reduction)."""
    return np.array([
        [float(sx), 0.0, 0.0],
        [0.0, float(sy), 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)


def matrice_shear_x(sh_x):
    """Matrice homogene 3x3 de transvection horizontale (shear en X)."""
    return np.array([
        [1.0, float(sh_x), 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)


def matrice_rotation(angle_deg):
    """Matrice homogene 3x3 de rotation en coordonnees image (axe y vers le bas).

    Avec y vers le bas, une rotation anti-horaire a l'ecran de angle_deg utilise
    +sin sur la 1re ligne et -sin sur la 2e ligne.
    """
    theta = np.radians(float(angle_deg))
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([
        [c, s, 0.0],
        [-s, c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)


def transformer_points(points, matrice):
    """Applique une matrice homogene 3x3 a un tableau de points 2D (N, 2)."""
    pts = np.asarray(points, dtype=float)
    un = np.ones((len(pts), 1), dtype=float)
    pts_homogenes = np.hstack([pts, un])
    pts_transformes = pts_homogenes @ matrice.T
    return pts_transformes[:, :2]


class TransformationModel:
    width = 1245
    height = 480

    points = np.array([
        [71, 136],  # RIGHT-HIP
        [63, 200],  # RIGHT-KNEE
        [40, 269],  # RIGHT-ANKLE
    ])

    connections = [(0, 1), (1, 2)]

    def __init__(self) -> None:
        self.image = None
        self.base_image_shape = None

        # Translation parameters
        self.trans_x = 0.0
        self.trans_y = 0.0

        # Scale parameters
        self.scale_x = 1.0
        self.scale_y = 1.0

        # Shear parameters
        self.shear = 1.0

        # Rotation parameters (90.0 correspond a la valeur affichee par defaut dans la vue)
        self.rot_angle = 90.0


    def update_param(self, value, param):
        assert hasattr(self, param)
        # Les QLineEdit envoient une chaine (parfois vide "" ou "-" pendant la saisie)
        if isinstance(value, str):
            texte = value.strip().replace(",", ".")
            if texte in ("", "-", "+", "."):
                return
            try:
                value = float(texte)
            except ValueError:
                return
        setattr(self, param, float(value))


    def set_image(self, image):
        self.image = np.ones((self.height, self.width, 3), dtype=np.uint8) * np.array([255, 255, 255], dtype=np.uint8)
        h, w, _ = image.shape
        if h >= self.height:
            h = self.height
        if w >= self.width:
            w = self.width
        self.image[:h, :w] = image[:h, :w]
        self.base_image_shape = (h, w)
        image = self.image.copy()
        image = self.draw_points(image, self.points)
        return image


    def draw_points(self, image, points):
        # /!\ Attention à modifier uniquement image et pas self.image!
        pts_int = np.round(points).astype(int)

        # On trace d'abord les segments orange puis les points jaunes par-dessus
        for idx1, idx2 in self.connections:
            p1 = (int(pts_int[idx1, 0]), int(pts_int[idx1, 1]))
            p2 = (int(pts_int[idx2, 0]), int(pts_int[idx2, 1]))
            cv2.line(image, p1, p2, COULEUR_CONNEXION_KNEEKG, EPAISSEUR_CONNEXION)

        for x, y in pts_int:
            cv2.circle(image, (int(x), int(y)), RAYON_POINT, COULEUR_POINT_KNEEKG, -1)

        return image


    def apply_transformation(self, transformation:str):
        assert transformation in ('translate', 'scale', 'shear', 'rotate')
        if self.image is None:
            return None

        print(f"Applying {transformation}")

        image = self.image.copy() # <- /!\ Modifier image et pas self.image
        points = self.points.copy()

        if transformation == 'translate':
            matrice = matrice_translation(self.trans_x, self.trans_y)
        elif transformation == 'scale':
            matrice = matrice_scale(self.scale_x, self.scale_y)
        elif transformation == 'shear':
            matrice = matrice_shear_x(self.shear)
        elif transformation == 'rotate':
            # On ramene le centre de l'image de base en (0, 0), on tourne, puis on place au centre du canevas
            if self.base_image_shape is not None:
                h_base, w_base = self.base_image_shape
            else:
                h_base, w_base = self.height, self.width

            t_vers_origine = matrice_translation(-w_base / 2.0, -h_base / 2.0)
            r_angle = matrice_rotation(self.rot_angle)
            t_vers_centre = matrice_translation(self.width / 2.0, self.height / 2.0)
            matrice = t_vers_centre @ r_angle @ t_vers_origine

        image = cv2.warpAffine(
            image,
            matrice[:2, :],
            (self.width, self.height),
            borderValue=COULEUR_FOND,
        )
        points = transformer_points(points, matrice)
        
        image = self.draw_points(image, points)
        return image
