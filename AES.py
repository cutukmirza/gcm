from Sboxes import Sbox
from Sboxes import Sbox_inv
import numpy as np

#************IMPLEMENTATION*********************************

rc = ['','01','02','04','08','10','20','40','80','1B','36']

fixed_matrix = [['02', '03', '01', '01'], 
['01', '02', '03', '01'], 
['01', '01', '02', '03'], 
['03', '01', '01', '02']]

fixed_matrix_inv = [['0E','0B', '0D','09'], 
['09', '0E', '0B', '0D'],
['0D', '09', '0E', '0B'],
['0B', '0D', '09', '0E']]


#function to copy a list
def Clone(li1): 
    li_copy = li1[:] 
    return li_copy 

#convert hex value to integer value
def hex_to_int(coor):
    new_val1 = "0x" + coor[0]
    out = int(new_val1, 16) 
    return out

#get value from sbox
def get_from_sbox (word):
    new_val = []
    for x in range(4):
        res = get_from_tuple(hex_to_int(word[x][0]), hex_to_int(word[x][1]))
        new_val.append(res[2:])
    return new_val

#get value from sbox_inv
def get_from_sbox_inv (word):
    new_val = []
    for x in range(4):
        res = get_from_tuple_inv(hex_to_int(word[x][0]), hex_to_int(word[x][1]))
        new_val.append(res[2:])
    return new_val

#function to get element from tuple using coordinates
def get_from_tuple(x, y):
    index = Sbox[y + (16 * x)]
    return hex(index)

#get elements from sbox_inv tuple using coordinates
def get_from_tuple_inv(x, y):
    index = Sbox_inv[y + (16 * x)]
    return hex(index)

#function to xor elements of two lists of length 4
def xor(list1, list2):
    l = []
    l.append(convert_to_hex2(list1[0])^convert_to_hex2(list2[0]))
    l.append(convert_to_hex2(list1[1])^convert_to_hex2(list2[1]))
    l.append(convert_to_hex2(list1[2])^convert_to_hex2(list2[2]))
    l.append(convert_to_hex2(list1[3])^convert_to_hex2(list2[3]))
    for x in range(len(l)):
        le = len (str(hex(l[x])[2:]))
        if (le == 1):
            l[x] = "0" + str(hex(l[x])[2:])
        else:
            l[x] = str(hex(l[x])[2:])
    return l

#function to xor elements of two lists of any length
def xor1(list1, list2):
    l = []
    for x in range(len(list1)):
        l.append(convert_to_hex2(list1[x])^convert_to_hex2(list2[x]))
    for x in range(len(l)):
        le = len (str(hex(l[x])[2:]))
        if (le == 1):
            l[x] = "0" + str(hex(l[x])[2:])
        else:
            l[x] = str(hex(l[x])[2:])
    return l

#function for polynomial multiplication
def galois_mult(a, b):
    p = 0
    hiBitSet = 0
    for _ in range(8):
        if b & 1 == 1:
            p ^= a
        hiBitSet = a & 0x80
        a <<= 1
        if hiBitSet == 0x80:
            a ^= 0x1b
        b >>= 1
    return p % 256

#matrix multiplication using polynomial multiplication of elements
def matr_mult(some_matrix):
    new_matrix = []
    for x in range(len(some_matrix)):
        f = 0
        while(f < 4):
            item_list = []
            for y in range(len(some_matrix)):
                item = galois_mult(convert_to_hex2(fixed_matrix[x][y]), convert_to_hex2(some_matrix[y][f]))
                item_list.append(item)
            item_list_xor = item_list[0]^item_list[1]^item_list[2]^item_list[3]    
            if (len(hex(item_list_xor)[2:]) == 1):
                new_matrix.append('0' + hex(item_list_xor)[2:])
            else:  
                new_matrix.append(hex(item_list_xor)[2:])
            f += 1
    return new_matrix 

#matrix multiplication using polynomial multiplication of elements
def matr_mult_inv(some_matrix):
    new_matrix = []
    for x in range(len(some_matrix)):
        f = 0
        while(f < 4):
            item_list = []
            for y in range(len(some_matrix)):
                item = galois_mult(convert_to_hex2(fixed_matrix_inv[x][y]), convert_to_hex2(some_matrix[y][f]))
                item_list.append(item)
            item_list_xor = item_list[0]^item_list[1]^item_list[2]^item_list[3]    
            if (len(hex(item_list_xor)[2:]) == 1):
                new_matrix.append('0' + hex(item_list_xor)[2:])
            else:  
                new_matrix.append(hex(item_list_xor)[2:])
            f += 1
    return new_matrix 

#bitwise rotation left function
def rotation(word):
    rotated = Clone(word)
    temp = rotated [0]
    for x in range (len(rotated)):
        if (x == len(rotated) - 1):
            rotated[x] = temp
        else:
            rotated[x] = rotated[x+1]
    return rotated

#bitwise rotation right function
def rotation_right(word):
    rotated = Clone(word)
    temp = rotated [len(rotated) - 1]
    for x in reversed(range (len(rotated))):
        if (x == 0):
            rotated[x] = temp
        else:
            rotated[x] = rotated[x - 1] 
    return rotated

#shift row left by one element, used in the function above
def shift_row_left(word, degree):
    for _ in range(degree):
        word = rotation(word) 
    return word

#shift row right by one element, used in the function above
def shift_row_right(word, degree):
    for _ in range(degree):
        word = rotation_right(word) 
    return word


#function to convert a string to hex
def convert_to_hex(some_string):
    string_hex = []
    for c in some_string:
        string_hex.append(hex(ord(c))[2:])
    return string_hex

#converts a hex string to an integer
def convert_to_hex2(s_string):
    hex_val = "0x" + s_string
    an_integer = int(hex_val, 16)
    return an_integer

#function to split ANY list into chunks of 4 elements
#the name is because it started as only a key splitter
#but ended up being used for everything else
def split_key(key):
    byte_list = []
    key_list = []
    for x in range (len(key)):
        byte_list.append(key[x])
        if (x % 4 == 3):
            key_list.append(byte_list)
            byte_list = []
    return key_list

#expand the key based on the rules from the tp pdf
def key_expansion(key_word_list, N, R):
    word_list = []
    for k in range(R * 4):
        if (k < N):
            word_list.append(key_word_list[k])
        elif(k >= N and k % N == 0 % N):
            res = xor(xor(word_list[k - N], get_from_sbox(rotation(word_list[k - 1]))), [str(rc[k//N]), "00", "00", "00" ])
            word_list.append(res)
        elif(k >= N and N > 6 and k % N == 4 % N):
            res = word_list[k - N] ^ get_from_sbox(rotation(word_list[k - 1]))
            word_list.append(res)
        else:
            res = xor(word_list[k - N], word_list[k - 1])
            word_list.append(res)
    return word_list

def AES_encrypt(key, plaintext):
    #key in hex
    key_hex = convert_to_hex(key)
    #print(key_hex)
    N = 0
    R = 0

    if (len(key) == 16):
        N = 4
        R = 11
    elif (len(key) == 24):
        N = 6
        R = 13
    elif (len(key) == 32):
        N = 8
        R = 15

    #plaintext 
    plain = plaintext
    #plaintext in hex
    #see if the length of the plaintext is a multiple of 128
    #if not pad with 0s
    plain_hex = convert_to_hex(plain)
    if (len(plain_hex) % 16 != 0):
        while(len(plain_hex) % 16 != 0):
            plain_hex.append('00')
    
    #make 128 bit blocks of plaintext
    plaintext_list = []
    plaintext_128 = []
    for x in range(1, len(plain_hex)+1): 
        plaintext_128.append(plain_hex[x-1]) 
        if (x % 16 == 0):
            plaintext_list.append(plaintext_128)
            plaintext_128 = []

    #split the key into chunks of 4 bytes
    key_byte_list = (split_key(key_hex))
    #expand the keys (here, we obtain 
    #a list of words and divide it into keys later on)
    op = key_expansion(key_byte_list, N, R)

    #make a list of keys from the list of words (op)
    key_list = []
    key = []
    for x in range(1, len(op)+1): 
        key += op[x-1]
        if (x % 4 == 0):
            key_list.append(key)
            key = []
    
    cipher_list = []
    for plain_item in plaintext_list:
        #transpose the plaintext converted into hex
        state_matrix = []
        plain_byte_list = split_key(plain_item)
        numpy_array = np.array(plain_byte_list)
        transpose = numpy_array.T
        state_matrix = transpose.tolist()
        #transpose the first key
        key_matrix = []
        list_of_lists = [op[0], op[1], op[2], op[3]]
        numpy_array = np.array(list_of_lists)
        transpose = numpy_array.T
        key_matrix = transpose.tolist()

        #encription
        #step 1: xor state matrix with key
        for x in range(len(state_matrix)):
            result = xor(state_matrix[x], key_matrix[x])
            state_matrix[x] = result

        #step 2: Substitute bytes
        for x in range (len(state_matrix)):
            state_matrix[x] = get_from_sbox(state_matrix[x])

        #step 3: Start rounds
        for ite in range(1, len(key_list)):
            #Substitute byte
            if (ite != 1):
                for x in range (len(state_matrix)):
                    state_matrix[x] = get_from_sbox(state_matrix[x])
            #Shift row
            for x in range(len(state_matrix)):
                state_matrix[x] = shift_row_left(state_matrix[x], x)

            #Mix Column
            #not done on the last iteration
            if (ite != (len(key_list) - 1)):     
                state_matrix = matr_mult(state_matrix)
            #we have to convert the state matrix to a different
            #format since we are not doing the multiplication
            else:
                temp = state_matrix.copy()
                state_matrix = []
                for x in range(len(temp)): 
                    state_matrix += temp[x]
            #xor with current round key
            #transpose the key first
            key_split = split_key(key_list[ite])
            numpy_array = np.array(key_split)
            transpose = numpy_array.T
            key_transpose = transpose.tolist()
            key_transpose_unpacked = []
            for x in range(1, len(key_transpose)+1): 
                key_transpose_unpacked += key_transpose[x-1]

            #xor with key
            state_matrix = xor1(state_matrix, key_transpose_unpacked)
            state_matrix = split_key(state_matrix)
            #transpose the state matrix to get ciphertext
            numpy_array = np.array(state_matrix)
            transpose = numpy_array.T
            key_transpose = transpose.tolist()
            ciphertext = []
            for x in range(1, len(key_transpose)+1): 
                ciphertext += key_transpose[x-1]
        cipher_list.append(ciphertext)
    return cipher_list

def AES_decrypt(key, ciphertext):
    for element in ciphertext:
        #key in hex
        key_hex = convert_to_hex(key)
        #print(key_hex)
        N = 0
        R = 0

        if (len(key) == 16):
            N = 4
            R = 11
        elif (len(key) == 24):
            N = 6
            R = 13
        elif (len(key) == 32):
            N = 8
            R = 15

        #now plain is equal to ciphertext since we are decrypting 
        #ciphertext in hex
        #see if the length of the plaintext is a multiple of 128
        #if not pad with 0s
        plain_hex = element
        if (len(plain_hex) % 16 != 0):
            while(len(plain_hex) % 16 != 0):
                plain_hex.append('00')
        
        #make 128 bit blocks of ciphertext
        cipher_list = []
        cipher_128 = []
        for x in range(1, len(plain_hex)+1): 
            cipher_128.append(plain_hex[x-1]) 
            if (x % 16 == 0):
                cipher_list.append(cipher_128)
                cipher_128 = []

        #split the key into chunks of 4 bytes
        key_byte_list = (split_key(key_hex))
        #expand the keys (here, we obtain 
        #a list of words and divide it into keys later on)
        op = key_expansion(key_byte_list, N, R)

        #make a list of keys from the list of words (op)
        key_list = []
        key = []
        for x in range(1, len(op)+1): 
            key += op[x-1]
            if (x % 4 == 0):
                key_list.append(key)
                key = []
        plain_list = []
        for plain_item in cipher_list:
            #transpose the ciphertext converted into hex
            state_matrix = []
            plain_byte_list = split_key(plain_item)
            numpy_array = np.array(plain_byte_list)
            transpose = numpy_array.T
            state_matrix = transpose.tolist()
            #transpose the last key
            key_matrix = []
            list_of_lists = [op[len(op) - 4], op[len(op) - 3], op[len(op) - 2], op[len(op) - 1]]
            numpy_array = np.array(list_of_lists)
            transpose = numpy_array.T
            key_matrix = transpose.tolist()

            #decryption
            #step 1: xor state matrix with key
            for x in range(len(state_matrix)):
                result = xor(state_matrix[x], key_matrix[x])
                state_matrix[x] = result
            
            #step 2: Shift row
            for x in range(len(state_matrix)):
                state_matrix[x] = shift_row_right(state_matrix[x], x)
            
            #step 3: Substitute bytes
            for x in range (len(state_matrix)):
                state_matrix[x] = get_from_sbox_inv(state_matrix[x])
            
            

            
            key_list.reverse()
            #step 3: Start rounds
            for ite in range(1, len(key_list)):
                
                    #xor with current round key
                    #transpose the key first
                    key_split = split_key(key_list[ite])
                    numpy_array = np.array(key_split)
                    transpose = numpy_array.T
                    key_transpose = transpose.tolist()
                    key_transpose_unpacked = []
                    for x in range(1, len(key_transpose)+1): 
                        key_transpose_unpacked += key_transpose[x-1]

                    state_matrix_unpacked = []
                    for x in range(1, len(state_matrix)+1): 
                        state_matrix_unpacked += state_matrix[x-1]

                    #xor with key
                    state_matrix = xor1(state_matrix_unpacked, key_transpose_unpacked)
                    state_matrix = split_key(state_matrix)
                    #transpose the state matrix to get ciphertext
                    numpy_array = np.array(state_matrix)
                    transpose = numpy_array.T
                    key_transpose = transpose.tolist()
                    #state_matrix = []
                    #for x in range(1, len(key_transpose)+1): 
                    #    state_matrix += key_transpose[x-1]
                    if (ite != (len(key_list)-1)):

                        #Mix Column
                        #not done on the last iteration
                        
                        state_matrix = matr_mult_inv(state_matrix)
                        
                        state_matrix = split_key(state_matrix)
                        #Shift row
                        for x in range(len(state_matrix)):
                            state_matrix[x] = shift_row_right(state_matrix[x], x)
                        
                        #Substitute byte
                        
                        for x in range (len(state_matrix)):
                            state_matrix[x] = get_from_sbox_inv(state_matrix[x])
        
        #transpose the state matrix to get ciphertext
        numpy_array = np.array(state_matrix)
        transpose = numpy_array.T
        key_transpose = transpose.tolist()
        ciphertext = []
        for x in range(1, len(key_transpose)+1): 
            ciphertext += key_transpose[x-1]
        plain_list.append(ciphertext)
    print("decrypted message",plain_list)
    original_message = ""
    for item in plain_list:
        for x in item:
            hex_int = convert_to_hex2(x)
            original_message += str(chr(hex_int))
    return original_message
#**************IMPLEMENTATION END*****************************

#***************TESTING***************************************
#starting key
""" key = "Thats my Kung Fu"
plain = "Two One Nine Two"
print("encrypted message",AES_encrypt(key, plain))
decrypted_message = AES_decrypt(key, AES_encrypt(key, plain))
print(decrypted_message) """
