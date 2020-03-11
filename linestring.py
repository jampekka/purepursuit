import numpy as np
import scipy.interpolate
import numba

def linesegment_project(start, end, p):
    start = np.atleast_2d(start)
    end = np.atleast_2d(end)
    p = np.atleast_2d(p)
    v = end - start
    u = start - p
    
    uv = np.einsum('nd,nd->n', u, v)
    vv = np.einsum('nd,nd->n', v, v)
    
    t = np.zeros(len(uv))
    np.divide(uv, vv, where=vv > 0, out=t)
    t = np.clip(-t, 0, 1)
    
    return t

def linesegment_distance(start, end, p):
    p = np.array(p)
    t = linesegment_project(start, end, p).reshape(-1, 1)
    error = ((1 - t)*start + t*end) - p
    d = np.linalg.norm(error, axis=1)
    return d.reshape(-1)

@numba.njit
def _lineseg_project(s, e, p):
    v = e - s
    u = s - p
    
    vv = np.dot(v, v)
    if vv == 0.0:
        return 0.0, s
    
    uv = np.dot(u, v)

    t = -uv/vv
    t = min(1.0, max(0.0, t))
    pp = (1 - t)*s + t*e
    
    return t, pp

@numba.njit
def linestring_project(points, p):
    min_error = np.inf
    best_i = 0
    best_t = 0.0

    for i in range(len(points) - 1):
        s = points[i]
        e = points[i+1]
        
        t, pp = _lineseg_project(s, e, p)
        d = p - pp
        error = np.dot(d, d)
        if error < min_error:
            min_error = error
            best_i = i
            best_t = t
    
    return best_i, best_t

@numba.njit
def linestring_project_dist(points, dists, p):
    i, t = linestring_project(points, p)

    return dists[i] + t*(dists[i+1] - dists[i])

@numba.njit
def linestring_interpolate(points, dists, d):
    d = d%dists[-1]
    s = np.searchsorted(dists, d) - 1
    e = s + 1
    t = (d - dists[s])/(dists[e] - dists[s])
    return (1 - t)*points[s] + t*points[e]

def _do_simplify_linestring(points, threshold):
    if len(points) == 2:
        return points
    [start, *mid, end] = points

    dists = np.atleast_1d(linesegment_distance(start, end, mid))
    furthest = np.argmax(dists)
    d = float(dists[furthest])
    
    if d < threshold:
        return [start, end]
    
    furthest += 1
    before = _do_simplify_linestring(points[:furthest+1], threshold)[:-1]
    after = _do_simplify_linestring(points[furthest:], threshold)
    return before + after

def simplify_linestring(points, *args, **kwargs):
    points = list(points)
    points = _do_simplify_linestring(points, *args, **kwargs)
    return np.array(points)

def simplify_linestring_idx(points, threshold, si=0, ei=None):
    if ei is None:
        ei = len(points) - 1

    if ei - si < 2:
        return [si, ei]
    
    [start, *mid, end] = points[si:ei+1]

    dists = np.atleast_1d(linesegment_distance(start, end, mid))
    furthest = np.argmax(dists)
    d = float(dists[furthest])
    
    if d < threshold:
        return [si, ei]
    
    furthest += si + 1
    before = simplify_linestring_idx(points, threshold, si, furthest)[:-1]
    after = simplify_linestring_idx(points, threshold, furthest, ei)
    return before + after

def prune_linestring(points):
    points = np.array(points)
    s, m, e = points[:-2], points[1:-1], points[2:]
    
    a = np.linalg.norm(s - m, axis=1)
    b = np.linalg.norm(e - m, axis=1)
    c = np.linalg.norm(s - e, axis=1)
    S = (a + b + c)/2
    A = S*(S - a)*(S - b)*(S - c)
    
    valid = [0] + list(np.flatnonzero(A > 0)) + [len(points) - 1]
    return points[valid]

# TODO: Handle loops
class LineString:
    def __init__(self, points):
        self.points = np.array(points)
        self.lengths = np.linalg.norm(np.diff(self.points, axis=0), axis=1)

        self.dist = np.zeros(len(self.points))
        self.dist[1:] = np.cumsum(self.lengths)
        self.total_dist = self.dist[-1]

        self.project = np.vectorize(self.project_one, signature='(d)->()')
        self.interpolator = scipy.interpolate.interp1d(self.dist, self.points, axis=0,
                fill_value=(self.points[0], self.points[-1]), bounds_error=False)
    
    def interpolate(self, d):
        return self.interpolator(d)

    def project_one(self, p):
        #closest2 = np.argmin(linesegment_distance(self.points[:-1], self.points[1:], p))
        #t = linesegment_project(self.points[closest], self.points[closest+1], p)
        closest, t = linestring_project(self.points, p)
        #print(closest, closest2)
        #assert(closest == closest2)
        return self.dist[closest] + t*self.lengths[closest]

    def index_at(self, d):
        return self.dist.searchsorted(d) - 1
    
    def signed_error(self, d, p):
        d = np.atleast_1d(d)
        p = np.atleast_2d(p)
        idx = self.index_at(d)
        m = self.interpolate(d)
        
        error = np.linalg.norm(p - m, axis=1)
        start, end = self.points[idx], self.points[idx+1]
        d = (p[:,0] - start[:,0])*(end[:,1] - start[:,1]) - (p[:,1] - start[:,1])*(end[:,0] - start[:,0])

        return error*np.sign(d)

class LineLoop(LineString):
    def interpolate(self, d):
        return self.interpolator(d%self.total_dist)

def extrude_linestring(path, width, direction=None):
    if direction is None:
        dx = np.gradient(path[:,0])
        dy = np.gradient(path[:,1])
        direction = np.array([dx, dy]).T
        direction /= np.linalg.norm(direction, axis=1).reshape(-1, 1)
        direction = direction[:,::-1]
        direction[:,1] *= -1

    return path + direction*width
