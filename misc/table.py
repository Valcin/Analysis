def table_write_pl(a1, b1, c1, d1, e1, f1, g1, h1, a2, b2, c2, d2, e2, f2, g2, h2,  a3, b3, c3, d3, e3, f3, g3, h3,\
a4, b4, c4, d4, e4, f4, g4, h4, name):
	
	a1 = round(a1, 3)
	b1 = round(b1, 3)
	c1 = round(c1, 3)
	d1 = round(d1, 3)
	e11 = round(e1[1], 3); e12 = round(e1[2], 3)
	f11 = round(f1[1], 3); f12 = round(f1[2], 3)
	g11 = round(g1[1], 3); g12 = round(g1[2], 3)
	h11 = round(h1[1], 3); h12 = round(h1[2], 3)
	a2 = round(a2, 3)
	b2 = round(b2, 3)
	c2 = round(c2, 3)
	d2 = round(d2, 3)
	e21 = round(e2[1], 3); e22 = round(e2[2], 3)
	f21 = round(f2[1], 3); f22 = round(f2[2], 3)
	g21 = round(g2[1], 3); g22 = round(g2[2], 3)
	h21 = round(h2[1], 3); h22 = round(h2[2], 3)
	a3 = round(a3, 3)
	b3 = round(b3, 3)
	c3 = round(c3, 3)
	d3 = round(d3, 3)
	e31 = round(e3[1], 3); e32 = round(e3[2], 3)
	f31 = round(f3[1], 3); f32 = round(f3[2], 3)
	g31 = round(g3[1], 3); g32 = round(g3[2], 3)
	h31 = round(h3[1], 3); h32 = round(h3[2], 3)
	a4 = round(a4, 3)
	b4 = round(b4, 3)
	c4 = round(c4, 3)
	d4 = round(d4, 3)
	e41 = round(e4[1], 3); e42 = round(e4[2], 3)
	f41 = round(f4[1], 3); f42 = round(f4[2], 3)
	g41 = round(g4[1], 3); g42 = round(g4[2], 3)
	h41 = round(h4[1], 3); h42 = round(h4[2], 3)
	
	
	
	from astropy.io import ascii
	ascii.write([[a1,a2,a3,a4],[e11,e21,e31,e41],[e12,e22,e32,e42],\
	[b1,b2,b3,b4],[f11,f21,f31,f41],[f12,f22,f32,f42],\
	[c1,c2,c3,c4],[g11,g21,g31,g41],[g12,g22,g32,g42],\
	[d1,d2,d3,d4],[h11,h21,h31,h41],[h12,h22,h32,h42]],name, format='latex',\
	 names = ['b1', 'errb1 -','errb1 +', 'b2', 'errb2 -', 'errb2 +', 'b3', 'errb3 -', 'errb3 +', 'b4', 'errb4 -', 'errb4 +'])  
	
	return
	
def table_write_pt2(a1, b1, c1, e1, f1, g1, a2, b2, c2, e2, f2, g2,  a3, b3, c3,  e3, f3, g3, \
a4, b4, c4,  e4, f4, g4, name):
	
	a1 = round(a1, 3)
	b1 = round(b1, 3)
	c1 = round(c1, 3)
	#~ d1 = round(d1, 3)
	e11 = round(e1[1], 3); e12 = round(e1[2], 3)
	f11 = round(f1[1], 3); f12 = round(f1[2], 3)
	g11 = round(g1[1], 3); g12 = round(g1[2], 3)
	#~ h11 = round(h1[1], 3); h12 = round(h1[2], 3)
	a2 = round(a2, 3)
	b2 = round(b2, 3)
	c2 = round(c2, 3)
	#~ d2 = round(d2, 3)
	e21 = round(e2[1], 3); e22 = round(e2[2], 3)
	f21 = round(f2[1], 3); f22 = round(f2[2], 3)
	g21 = round(g2[1], 3); g22 = round(g2[2], 3)
	#~ h21 = round(h2[1], 3); h22 = round(h2[2], 3)
	a3 = round(a3, 3)
	b3 = round(b3, 3)
	c3 = round(c3, 3)
	#~ d3 = round(d3, 3)
	e31 = round(e3[1], 3); e32 = round(e3[2], 3)
	f31 = round(f3[1], 3); f32 = round(f3[2], 3)
	g31 = round(g3[1], 3); g32 = round(g3[2], 3)
	#~ h31 = round(h3[1], 3); h32 = round(h3[2], 3)
	a4 = round(a4, 3)
	b4 = round(b4, 3)
	c4 = round(c4, 3)
	#~ d4 = round(d4, 3)
	e41 = round(e4[1], 3); e42 = round(e4[2], 3)
	f41 = round(f4[1], 3); f42 = round(f4[2], 3)
	g41 = round(g4[1], 3); g42 = round(g4[2], 3)
	#~ h41 = round(h4[1], 3); h42 = round(h4[2], 3)
	
	
	
	from astropy.io import ascii
	ascii.write([[a1,a2,a3,a4],[e11,e21,e31,e41],[e12,e22,e32,e42],\
	[b1,b2,b3,b4],[f11,f21,f31,f41],[f12,f22,f32,f42],\
	[c1,c2,c3,c4],[g11,g21,g31,g41],[g12,g22,g32,g42]],name, format='latex',\
	 names = ['b1', 'errb1 -','errb1 +', 'b2', 'errb2 -', 'errb2 +', 'bs', 'errbs -', 'errbs +'])  
	
	return
	
def table_write_pt3(a1, b1, c1, d1, e1, f1, g1, h1, a2, b2, c2, d2, e2, f2, g2, h2,  a3, b3, c3, d3, e3, f3, g3, h3,\
a4, b4, c4, d4, e4, f4, g4, h4, name):
	
	a1 = round(a1, 3)
	b1 = round(b1, 3)
	c1 = round(c1, 3)
	d1 = round(d1, 3)
	e11 = round(e1[1], 3); e12 = round(e1[2], 3)
	f11 = round(f1[1], 3); f12 = round(f1[2], 3)
	g11 = round(g1[1], 3); g12 = round(g1[2], 3)
	h11 = round(h1[1], 3); h12 = round(h1[2], 3)
	a2 = round(a2, 3)
	b2 = round(b2, 3)
	c2 = round(c2, 3)
	d2 = round(d2, 3)
	e21 = round(e2[1], 3); e22 = round(e2[2], 3)
	f21 = round(f2[1], 3); f22 = round(f2[2], 3)
	g21 = round(g2[1], 3); g22 = round(g2[2], 3)
	h21 = round(h2[1], 3); h22 = round(h2[2], 3)
	a3 = round(a3, 3)
	b3 = round(b3, 3)
	c3 = round(c3, 3)
	d3 = round(d3, 3)
	e31 = round(e3[1], 3); e32 = round(e3[2], 3)
	f31 = round(f3[1], 3); f32 = round(f3[2], 3)
	g31 = round(g3[1], 3); g32 = round(g3[2], 3)
	h31 = round(h3[1], 3); h32 = round(h3[2], 3)
	a4 = round(a4, 3)
	b4 = round(b4, 3)
	c4 = round(c4, 3)
	d4 = round(d4, 3)
	e41 = round(e4[1], 3); e42 = round(e4[2], 3)
	f41 = round(f4[1], 3); f42 = round(f4[2], 3)
	g41 = round(g4[1], 3); g42 = round(g4[2], 3)
	h41 = round(h4[1], 3); h42 = round(h4[2], 3)
	
	
	
	from astropy.io import ascii
	ascii.write([[a1,a2,a3,a4],[e11,e21,e31,e41],[e12,e22,e32,e42],\
	[b1,b2,b3,b4],[f11,f21,f31,f41],[f12,f22,f32,f42],\
	[c1,c2,c3,c4],[g11,g21,g31,g41],[g12,g22,g32,g42],\
	[d1,d2,d3,d4],[h11,h21,h31,h41],[h12,h22,h32,h42]],name, format='latex',\
	 names = ['b1', 'errb1 -','errb1 +', 'b2', 'errb2 -', 'errb2 +', 'bs', 'errbs -', 'errbs +', 'b3nl', 'errb3nl -', 'errb3nl +'])  
	
	return
