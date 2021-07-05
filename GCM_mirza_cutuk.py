from AES import AES_encrypt
from AES import AES_decrypt
from AES import xor1 as xor
from AES import convert_to_hex
from functools import reduce
import numpy as np
from textwrap import wrap

#************IMPLEMENTATION*********************************

#converts a hex string to an integer
def hex_to_int(hex_string):
    hex_val = "0x" + hex_string
    an_integer = int(hex_val, 16)
    return an_integer

#function to convert a string to hex
def str_to_hex(some_string):
    string_hex = []
    for c in some_string:
        string_hex.append(hex(ord(c))[2:])
    return string_hex

#function to convert hex to ascii string
def hex_to_str(hex_string):
    bytes_object = bytes.fromhex(hex_string)
    ascii_string = bytes_object.decode("ascii")
    return ascii_string

#convert a list of hex values into a string
def hexlist_to_str(hex_list):
    hex_str = ""
    for element in hex_list:
        hex_int = hex_to_int(element)
        hex_str += str(chr(hex_int))
    return hex_str

#function to split a list into lists of size 16
def split_128(l):
    s_l = []
    res = []
    counter = 0
    for item in l:
        res.append(item)
        counter += 1
        if (counter % 16 == 0):
            s_l.append(res)
            res = []
        if (counter == len(l) and counter % 16 != 0):
            s_l.append(res)
    return s_l

# convert a list into a matrix    
def list_to_matrix(l):
    matrix = []
    counter = 0
    res = []
    for item in l:
        res.append(item)
        counter += 1
        if (counter % 4) == 0:
            matrix.append(res)
            res = []

    # transpose the list   
    numpy_array = np.array(matrix)
    transpose = numpy_array.T
    transpose_list = list(transpose)
    return transpose_list

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




# galois field authentication method
def gal_aut(A, C, L, H):
    # take the first block IV1 and multiply by H
    res = matrix_mul(A[0], H)
    # xor that result with IVn and multiply by H
    # if there are more than 1 blocks of A
    for i in range(1, len(A)):
        res = xor(res, A[i])
        res = matrix_mul(res, H)

    for i in range(len(C)):
        res = xor(res, C[i])
        res = matrix_mul(res, H)
    # when we are finished with A_n and C_n 
    # xor the result with L
    res = xor(res, L)
    # and then multiply by H
    res = matrix_mul(res, H)
    return res

#galois multiplication of two matrices of same size
def matrix_mul(l1, l2):
    m1 = list_to_matrix(l1)
    m2 = list_to_matrix(l2)
    setGF2(128, 0b10000111)
    new_matrix = []
    for x in range(len(m1)):
        f = 0
        while(f < 4):
            item_list = []
            for y in range(len(m1)):
                item = multGF2(hex_to_int(m2[x][y]), hex_to_int(m1[y][f]))                
                item_list.append(item % 256)

            item_list_xor = item_list[0]^item_list[1]^item_list[2]^item_list[3]    
            if (len(hex(item_list_xor)[2:]) == 1):
                new_matrix.append('0' + hex(item_list_xor)[2:])
            else:  
                new_matrix.append(hex(item_list_xor)[2:])
            f += 1
    return new_matrix 

# compute the L_IV
def compute_L_IV(IV):
    l_iv = str(len(IV))
    l_iv_hex = str_to_hex(l_iv)
    res = []
    # add 0^64 0s to result
    for _ in range(8):
        res.append('00')
    # add 0s in front of l_iv if necessary
    for _ in range(8 - len(l_iv_hex)):
        res.append('00')
    # add each element of l_iv_hex to the result
    for element in l_iv_hex:
        res.append(element)
    return res

def padding(val):
    val_len = len(val)
    for _ in range(16 - val_len):
        val.append('00')
    return val

# L = A_len || C_len
def get_L(A_len, C_len):

    # convert A_len and C_len to hex
    A_len_hex = hex(A_len)[2:]
    C_len_hex = hex(C_len)[2:]
    A_len_hex = wrap(A_len_hex, 2)
    C_len_hex = wrap(C_len_hex, 2)
    # make both these hex values into 64 bit size
    # save the size
    temp = A_len_hex.copy()
    A_len_hex = []
    # append 0s
    for _ in range(8 - len(temp)):
        A_len_hex.append('00')
    # add the length as last bits
    for item in temp:
        A_len_hex.append(item)

    temp = C_len_hex.copy()
    C_len_hex = []
    # append 0s
    for _ in range(8 - len(temp)):
        C_len_hex.append('00')
    # add the length as last bits
    for item in temp:
        C_len_hex.append(item)

    # merge the two lists to obtain L
    for item in C_len_hex:
        A_len_hex.append(item)
    
    L = A_len_hex.copy()
    return L



def calculate_initial_counter(IV, H, L_IV):
    empty_s = ['00', '00', '00', '00', '00', '00', '00', '00', '00', '00', '00', '00', '00', '00', '00', '00']
    # convert IV into hex
    IV_hex_list = convert_to_hex(IV)
    IV_hex = split_128(IV_hex_list)
    # check the last block of IV to see if it has length 128
    # if not padd with 0s
    if (len(IV_hex[len(IV_hex) - 1]) != 16):
        IV_hex[len(IV_hex) - 1] = padding(IV_hex[len(IV_hex) - 1])

    # if IV length is exactly 96 bits
    # then
    init_counter = [] 
    if (len(IV) == 12):
        for _ in range(3):
            IV_hex.append('00')
        IV_hex.append('01')
        init_counter = IV_hex.copy()
    else:
        # if IV has just one block
        if (len(IV_hex) == 1):
            # multiply empty by H
            res = matrix_mul(empty_s, H)
            # xor the result with IV_hex[0]
            res = xor(res, IV_hex[0])
            # multiply block by H
            res = matrix_mul(res, H)
            # xor that result with L_IV
            res = xor(res, L_IV)
            # set initial counter to this res
            init_counter = res
        #if IV has more than one block
        else:
            res = gal_aut(empty_s, IV_hex, L_IV, H)
            init_counter = res

    return init_counter

def counter_mode(key, C_start, plain_blocks):
    # start counter mode 
    cipher_list = []
    counter = C_start.copy()
    for block in plain_blocks:
        # convert counter to integer
        # first concatenate entire C_start list
        cs_concat = ""
        for item in counter[0]:
            cs_concat += item
        # convert that entire hex number into int
        counter = hex_to_int(cs_concat)
        
        # increment counter
        counter += 1
        # return counter to hex
        counter = hex(counter)
        # convert hex_string to regular string so it can be encrypted
        c_list = wrap(counter, 2)
        c_list.pop(0)
        counter = hexlist_to_str(c_list)
        # encrypt the counter with AES
        counter = AES_encrypt(key, counter)
        # if length of the last block is less than 128 bits
        # then we xor it with the first len(block) most significant bits
        if (len(block) < 16):
            temp_counter = []
            for elem in range(len(block)):
                temp_counter.append(counter[0][elem])
            cipher = xor(temp_counter, block)
        # otherwise we just do a xor with the entire counter    
        else:
            # xor with block
            cipher = xor(counter[0], block)
        # append to cipher_list
        cipher_list.append(cipher)
    return cipher_list

def counter_mode_decrypt(key, C_start, c_list):
    # start counter mode 
    plain_list = []
    counter = C_start.copy()
    for block in c_list:
        # convert counter to integer
        # first concatenate entire C_start list
        cs_concat = ""
        for item in counter[0]:
            cs_concat += item
        # convert that entire hex number into int
        counter = hex_to_int(cs_concat)
        
        
        # increment counter
        counter += 1
        # return counter to hex
        counter = hex(counter)
        # convert hex_string to regular string so it can be encrypted
        c_list = wrap(counter, 2)
        c_list.pop(0)
        counter = hexlist_to_str(c_list)
        # encrypt the counter with AES
        counter = AES_encrypt(key, counter)
        # if length of the last block is less than 128 bits
        # then we xor it with the first len(block) most significant bits
        if (len(block) < 16):
            temp_counter = []
            for elem in range(len(block)):
                temp_counter.append(counter[0][elem])
            plain = xor(temp_counter, block)
        # otherwise we just do a xor with the entire counter    
        else:
            # xor with block
            plain = xor(counter[0], block)
        # append to cipher_list
        plain_list.append(plain)
    return plain_list


def get_tag(A, c_list, C_start, H):
    # convert authenticated data to hex
    A_hex = str_to_hex(A) 
    # split authenticated data into blocks of 128 bits
    A_blocks = split_128(A_hex)
    # save the original length of authenticated data for later
    # take all the len(A_blocks) - 1, multiply by 16, and then add to it the length of the last block
    # since the last block size can vary
    A_len = (len(A_blocks) - 1) * 16 + len(A_blocks[len(A_blocks) - 1]) 
    # check the last block to see if it is 128 bits, if not, add padding of 0s
    if (len(A_blocks[len(A_blocks) - 1]) != 16):
        A_blocks[len(A_blocks) - 1] = padding(A_blocks[len(A_blocks) - 1])

    # cipher text is already in hex and split into blocks of 128 (last one can be less)
    # so we need to add padding to the cipher text as well
    # but first we save the original length to compute L
    # same process as for A_blocks
    C_len = (len(c_list) - 1) * 16 + len(c_list[len(c_list) - 1])
    # check the last cipher text block to see if it is 128 bits, if not add padding of 0s
    if (len(c_list[len(c_list) - 1]) != 16):
        c_list[len(c_list) - 1] = padding(c_list[len(c_list) - 1])

    # compute L
    L = get_L(A_len, C_len)

    # get authentication tag T
    # perform galois authentication for A and C (xor with L is inside the function)
    res = gal_aut(A_blocks, c_list, L, H)
    # xor the obtained result with C_start
    # to obtain T
    T = xor(res, C_start[0])
    
    #return tag T and cipher texts
    return T

def gcm_encrypt(P, IV, K, A):
    # Step 1
    # Compute Hashkey H, encrypt a 128bit message full of zeros with AES
    empty_128b = "0000000000000000"
    H = AES_encrypt(K, empty_128b)[0]
    
    # Step 2
    # Get value of the initial counter
    # compute L_IV
    L_IV = compute_L_IV(IV)
    # compute initial counter value
    init_counter = calculate_initial_counter(IV, H, L_IV)
    # convert initial counter to string so we could encrypt it with aes
    init_counter_str = hexlist_to_str(init_counter)

    # Step 3
    # Counter Mode

    # compute C_start by encrypting it with AES
    C_start = AES_encrypt(K, init_counter_str)

    # obtain ciphertext
    # Counter Mode
    # convert plaintext to hex
    plain_hex = str_to_hex(P)
    # split plaintext into blocks of 128 bits
    plain_blocks = split_128(plain_hex)
    # obtain cipher list by using counter mode
    c_list = counter_mode(K, C_start, plain_blocks)

    # obtain authentication tag
    T = get_tag(A, c_list, C_start, H)
    
    #return tag T and cipher texts
    return [T, c_list]

def gcm_decrypt(C, T, K, A, IV):
    c_list = C.copy()

    # Step 1
    # Compute Hashkey H, encrypt a 128bit message full of zeros with AES
    empty_128b = "0000000000000000"
    H = AES_encrypt(K, empty_128b)[0]

    L_IV = compute_L_IV(IV)
    # Compute initial counter value
    init_counter = calculate_initial_counter(IV, H, L_IV)
    init_counter_str = hexlist_to_str(init_counter)
    C_start = AES_encrypt(K, init_counter_str)
    # get tag T'
    T_p = get_tag(A, c_list, C_start, H)
    # if tag is different, that means our data was changed
    # so we just return Fail
    if (T != T_p):
        return "FAIL"
    else: 
        # get all the counters one by one to decrypt the cipher text
        # and decrypt the message by using counter mode decrypt function
        p_text_hex = counter_mode_decrypt(K, C_start, c_list)
        p_text = ""
        for block in p_text_hex:    
            p_text += str((hexlist_to_str(block)))
        return p_text

#**************IMPLEMENTATION END*****************************

#***************TESTING***************************************

# The plaintext P (any length)
P = "Two One Nine Two"
# The initialisation value IV 
IV = "Random IV genera"
# The Key K
K = "Thats my Kung Fu"
# Additional Authenticated Data A
A = "Random auth data"

print("Original message",P)
# test when T = T' *******************
print ("***test when T = T'***")
# encryption returns [T, c_list]
res = gcm_encrypt(P, IV, K, A)
C = res[1]
T = res[0]
print("Authentication tag: ", T)
print("Ciphertext: ", C)
# decryption
# decryption returns plaintext
dec_res = gcm_decrypt(C, T, K, A, IV)
print("Decrypted message: ", dec_res)
# test when T <> T' *******************
print ("***test when T <> T'***")
res = gcm_encrypt(P, IV, K, A)
# decryption returns plaintext
C = res[1]
print("original cyphertext: ", C)
T = res[0]
print("original tag: ", T)
# we change a couple of bytes in the cyphertext
C[0][4] = '3d'
C[0][10] = '00'
C[0][14] = '35'
print("changed cyphertext: ", C)
# decryption
dec_res = gcm_decrypt(C, T, K, A, IV)
# we should get fail
print("Decryption result: ", dec_res)


