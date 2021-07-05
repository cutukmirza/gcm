from functools import reduce
# constants used in the multGF2 function
mask1 = mask2 = polyred = None

def setGF2(degree, irPoly):
    # Define parameters of binary finite field GF(2^m)/g(x)
    # degree: if 2^m is our galois field, then degree = m
    # irPoly: coefficients of the irreducible polynomial g(x)
    
    # function to convert integer into a polynomial
    def int_to_poly(sInt):
        #Convert an integer into a polynomial
        return [(sInt >> i) & 1
                for i in reversed(range(sInt.bit_length()))]    
    
    global mask1, mask2, polyred
    mask1 = mask2 = 1 << degree
    mask2 -= 1
    polyred = reduce(lambda x, y: (x << 1) + y, int_to_poly(irPoly)[1:])
        
def multGF2(p1, p2):
    #Multiply two polynomials in GF(2^m)/g(x)
    p = 0
    while p2:
        if p2 & 1:
            p ^= p1
        p1 <<= 1
        if p1 & mask1:
            p1 ^= polyred
        p2 >>= 1
    return p & mask2

# testing part
# Define binary field GF(2^128)/x^7 + x^2 + x + 1
#setGF2(8, 0b100011011)

#setGF2(128, 0b10000111)
#print(multGF2(2,99))
#print("{:02x}".format(multGF2(2, 99)))