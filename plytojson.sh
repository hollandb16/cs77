#!/usr/bin/env python2.7

import os
import sys


if len(sys.argv) != 3:
    print('Usage: %s [plyfile] [jsonfile]' % sys.argv[0])
    sys.exit(1)

fn_ply  = sys.argv[1]
fn_json = sys.argv[2]

# read plyfile
ply = open(fn_ply,'rt').read().split('\n')
print('read %d lines from PLY file' % len(ply))

if ply[0] != 'ply':
    print('Error: expected "ply" on line 1')
    sys.exit(1)
if ply[1] != 'format ascii 1.0':
    print('Error: expected "format ascii 1.0" on line 2')

# PLY properties
n_v = 0
has_normal = False
has_uv = False
has_color = False
n_f = 0

# get PLY header
state = 'none'
i = 3
while ply[i] != 'end_header':
    l = ply[i]

    if l.startswith('comment '):
        i += 1
        continue

    if l.startswith('element vertex '):
        elem = l.split(' ')
        n_v = int(elem[2])
        state = 'vert'

    if state == 'vert' and l.startswith('property '):
        prop = l.split(' ')
        if prop[2] in ['x','y','z']:
            pass
        elif prop[2] in ['nx','ny','nz']:
            has_normal = True
        elif prop[2] in ['s','t']:
            has_uv = True
        elif prop[2] in ['red','green','blue']:
            has_color = True
        else:
            print('Error: unexpected vertex property on line %d: %s' % (i,l))


    if l.startswith('element face '):
        elem = l.split(' ')
        n_f = int(elem[2])
        state = 'face'

    if state == 'face' and l.startswith('property '):
        prop = l.split(' ')
        if prop[1] == 'list' and prop[4] == 'vertex_indices':
            pass
        else:
            print('Error: unexpected face property on line %d: %s' % (i,l))

    i += 1
if ply[i] == 'end_header':
  print('Ended header')
i += 1

iend_normal = 3 + (3 if has_normal else 0)
iend_uv     = iend_normal + (2 if has_uv else 0)
iend_color  = iend_uv + (3 if has_color else 0)

# get PLY vertex info
l_verts = []
for i_v in range(n_v):
    v = ply[i+i_v].split(' ')
    pos   = map(float,v[0:3])
    norm  = map(float,v[3:iend_normal]) if has_normal else None
    uv    = map(float,v[iend_normal:iend_uv]) if has_uv else None
    color = map(float,v[iend_uv:iend_color]) if has_color else None
    l_verts += [(pos,norm,uv,color)]
i += n_v

# get PLY face info
l_tris = []
l_quads = []
for i_f in range(n_f):
    f = ply[i+i_f].split(' ')
    if f[0] == '3':
        l_tris += [map(int,f[1:4])]
    elif f[0] == '4':
        l_quads += [map(int,f[1:5])]
    else:
        print('Error: unexpected face size: %s' % f[0])
        sys.exit(1)


print('count:\n  vert: %d\n  face: %d\n  tri: %d\n  quad: %d' % (n_v, n_f, len(l_tris),len(l_quads)))


# prep for JSON dump
l_data = ['"name": "%s"' % fn_ply]
# set default material
l_data += ['"material": { "ks": [0,0,0], "kd": [0.725,0.71,0.68], "n": 10.0 }']
# vertex info
l_data += ['"pos": [\n%s\n]' % (',\n'.join('%f,%f,%f' % (v[0][0],v[0][1],v[0][2]) for v in l_verts))]
if has_normal:
    l_data += ['"norm": [\n%s\n]' % (',\n'.join('%f,%f,%f' % (v[1][0],v[1][1],v[1][2]) for v in l_verts))]
if has_uv:
    l_data += ['"uv": [\n%s\n]' % (',\n'.join('%f,%f' % (v[2][0],v[2][1]) for v in l_verts))]
# triangle and quad info
if len(l_tris):
      l_data += ['"triangle": [\n%s\n]' % (',\n'.join('%d,%d,%d' % (t[0],t[1],t[2]) for t in l_tris))]
if len(l_quads):
    l_data += ['"quad": [\n%s\n]' % (',\n'.join('%d,%d,%d,%d' % (q[0],q[1],q[2],q[3]) for q in l_quads))]


# write to JSON
fp_json = open(fn_json,'wt')
fp_json.write('{\n')
fp_json.write(',\n'.join(l_data))
fp_json.write('}')
fp_json.close()
