import cv2
import numpy as np
from scipy.interpolate import CubicSpline


COULEUR_COURBE = (255, 255, 0)    # jaune (RGB)
COULEUR_POINT = (255, 0, 0)       # rouge (RGB)
COULEUR_VECTEUR = (0, 255, 0)     # vert (RGB)
EPAISSEUR_TRAIT = 2
RAYON_CENTRE_HERMITE = 4
NB_POINTS_COURBE = 100
FACTEUR_TANGENTE = 2.0            # echelle sur (p2 - p1) pour bien voir la tangence


def bezier_lineaire(p0, p1, t):
    """Courbe de Bezier a 2 points : B(t) = (1 - t)*P0 + t*P1."""
    u = 1.0 - t
    return u * p0 + t * p1


def bezier_quadratique(p0, p1, p2, t):
    """Courbe de Bezier a 3 points : B(t) = (1 - t)^2*P0 + 2*(1 - t)*t*P1 + t^2*P2."""
    u = 1.0 - t
    return (u ** 2) * p0 + 2.0 * u * t * p1 + (t ** 2) * p2


def bezier_cubique(p0, p1, p2, p3, t):
    """Courbe de Bezier a 4 points : B(t) = (1-t)^3*P0 + 3*(1-t)^2*t*P1 + 3*(1-t)*t^2*P2 + t^3*P3."""
    u = 1.0 - t
    return (
        (u ** 3) * p0
        + 3.0 * (u ** 2) * t * p1
        + 3.0 * u * (t ** 2) * p2
        + (t ** 3) * p3
    )


def bezier_casteljau(points, t):
    """Algorithme de De Casteljau pour n points de controle (utilise si n > 4)."""
    n = len(points)
    courant = np.repeat(points[None, :, :], len(t), axis=0)
    u = 1.0 - t[:, None, None]
    t_col = t[:, None, None]
    for k in range(1, n):
        courant = u * courant[:, :-1, :] + t_col * courant[:, 1:, :]
    return courant[:, 0, :]


def segment_hermite(m0, t0, m1, t1, t):
    """Segment d'Hermite entre les centres m0 et m1 avec les tangentes t0 et t1."""
    t2 = t ** 2
    t3 = t ** 3
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2
    return h00 * m0 + h10 * t0 + h01 * m1 + h11 * t1


class InterpolationModel:
    width = 1245
    height = 440

    point_radius = 6
    
    def __init__(self):
        self.curve_type = "Linear"
        self.reset()


    def reset_canvas(self):
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)


    def reset(self):
        self.points = []
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)


    def add_point(self, x, y):
        print(f"Adding point x={x} y={y}")
        self.points.append([x, y])
        cv2.circle(self.canvas, (round(x), round(y)), 5, COULEUR_POINT, thickness=-1)


    def set_curve_type(self, curve:str):
        print(f"Changed curve type to {curve}")
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.curve_type = curve


    def try_to_move_point(self, start_pt:tuple[int], end_pt:tuple[int]):
        print(f"Start = {start_pt} | End = {end_pt}")

        x, y = start_pt
        clicked_point_idx = None
        for idx, (xc, yc) in enumerate(self.points):
            if np.sqrt((x - xc)**2 + (y - yc)**2) < self.point_radius:
                clicked_point_idx = idx
                break
        
        if clicked_point_idx is not None:
            self.points[clicked_point_idx] = list(end_pt)


    def _tracer_polyligne(self, pts_courbe):
        """Trace une suite de points (N, 2) en jaune sur le canevas."""
        if pts_courbe is None or len(pts_courbe) < 2:
            return
        pts_int = np.round(pts_courbe).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(self.canvas, [pts_int], isClosed=False, color=COULEUR_COURBE, thickness=EPAISSEUR_TRAIT)

    
    def draw(self):
        self.reset_canvas()

        if self.curve_type == "Linear":
            self.draw_linear()
        elif self.curve_type == "Bezier":
            self.draw_bezier()
        elif self.curve_type == "Hermite":
            self.draw_hermite()
        elif self.curve_type == "BSpline":
            self.draw_spline()

        for x, y in self.points:
            cv2.circle(self.canvas, (round(x), round(y)), self.point_radius, COULEUR_POINT, -1)


    def draw_linear(self):
        """Relie chaque paire de points consecutifs par un segment de droite."""
        if len(self.points) < 2:
            return

        for i in range(len(self.points) - 1):
            p1 = (round(self.points[i][0]), round(self.points[i][1]))
            p2 = (round(self.points[i + 1][0]), round(self.points[i + 1][1]))
            cv2.line(self.canvas, p1, p2, COULEUR_COURBE, EPAISSEUR_TRAIT)


    def draw_bezier(self):
        """Trace une courbe de Bezier (lineaire, quadratique, cubique ou De Casteljau si > 4 points)."""
        n = len(self.points)
        if n < 2:
            return

        pts = np.asarray(self.points, dtype=float)
        t = np.linspace(0.0, 1.0, NB_POINTS_COURBE)[:, None]

        if n == 2:
            courbe = bezier_lineaire(pts[0], pts[1], t)
        elif n == 3:
            courbe = bezier_quadratique(pts[0], pts[1], pts[2], t)
        elif n == 4:
            courbe = bezier_cubique(pts[0], pts[1], pts[2], pts[3], t)
        else:
            courbe = bezier_casteljau(pts, t[:, 0])

        self._tracer_polyligne(courbe)
        return courbe


    def draw_hermite(self):
        """Construit un vecteur tangent vert par paire de clics et trace la courbe d'Hermite entre les centres."""
        nb_paires = len(self.points) // 2
        if nb_paires == 0:
            return

        centres = []
        tangentes = []

        for k in range(nb_paires):
            p1 = np.asarray(self.points[2 * k], dtype=float)
            p2 = np.asarray(self.points[2 * k + 1], dtype=float)

            # Segment vert reliant les deux clics du vecteur
            pt1 = (round(p1[0]), round(p1[1]))
            pt2 = (round(p2[0]), round(p2[1]))
            cv2.line(self.canvas, pt1, pt2, COULEUR_VECTEUR, EPAISSEUR_TRAIT)

            # Milieu du segment = point d'ancrage de la courbe d'Hermite
            milieu = 0.5 * (p1 + p2)
            tangente = FACTEUR_TANGENTE * (p2 - p1)
            centres.append(milieu)
            tangentes.append(tangente)

        # Dès qu'on a au moins 2 vecteurs, on relie leurs centres avec les polynomes d'Hermite
        if len(centres) >= 2:
            t = np.linspace(0.0, 1.0, NB_POINTS_COURBE)[:, None]
            for i in range(len(centres) - 1):
                courbe_i = segment_hermite(centres[i], tangentes[i], centres[i + 1], tangentes[i + 1], t)
                self._tracer_polyligne(courbe_i)

        # On dessine le petit point rouge au centre de chaque vecteur vert
        for m in centres:
            cv2.circle(self.canvas, (round(m[0]), round(m[1])), RAYON_CENTRE_HERMITE, COULEUR_POINT, -1)


    def draw_spline(self):
        """Trace une spline cubique de degre 3 avec scipy.interpolate.CubicSpline."""
        n = len(self.points)
        if n < 2:
            return

        pts = np.asarray(self.points, dtype=float)
        if n == 2:
            self.draw_linear()
            return

        # Parametrisation par la distance cumulee u pour pouvoir cliquer dans n'importe quel sens
        diffs = np.diff(pts, axis=0)
        pas = np.maximum(np.hypot(diffs[:, 0], diffs[:, 1]), 1e-6)
        u = np.concatenate(([0.0], np.cumsum(pas)))

        spline = CubicSpline(u, pts, bc_type="natural")
        u_fin = np.linspace(0.0, u[-1], NB_POINTS_COURBE * (n - 1) // 2)
        courbe = spline(u_fin)
        self._tracer_polyligne(courbe)
        
    def get_canvas(self):
        self.draw()
        return self.canvas