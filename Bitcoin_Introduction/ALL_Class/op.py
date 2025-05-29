from ALL_Class.Module import *
from ALL_Class.Helper import *

def op_0(stack : list):
    stack.append(encode_num(0))    
    return True 

def op_1(stack : list ):
    stack.append(encode_num(1))    
    return True 

def op_2(stack:list):
    stack.append(encode_num(2))    
    return True 

def op_dup(stack : list ):
    """
    Duplicates the top element of the stack.
    """
    if len(stack) < 1 :
        return False
    stack.append(stack[-1])
    return True 

def op_hash160(stack : list ):
    """
    Hashes the top element of the stack using SHA-256 followed by RIPEMD-160.
    """
    if len(stack) < 1:
        return False
    top_element = stack.pop()
    hash_result = hash160(top_element)
    stack.append(hash_result)
    return True

def op_hash256(stack : list):
    """
    Hashes the top element of the stack using SHA-256 twice.
    """
    if len(stack) < 1:
        return False
    element = stack.pop()
    stack.append(hash256(element))
    return True

def op_ripemd160(stack : list ):
    """
    Hashes the top element of the stack using RIPEMD-160.
    """
    if len(stack) < 1:
        return False
    top_element = stack.pop()
    hash_result = hashlib.new('ripemd160', top_element).digest()
    stack.append(hash_result)
    return True

def op_sha256(stack : list ):
    """
    Hashes the top element of the stack using SHA-256.
    """
    if len(stack) < 1:
        return False
    top_element = stack.pop()
    hash_result = hashlib.sha256(top_element).digest()
    stack.append(hash_result)
    return True

def op_6(stack : list ):
    stack.append(encode_num(6))
    return True

def op_equal(stack):
    if len(stack) < 2:
        return False
    a = stack.pop()
    b = stack.pop()
    if a == b:
        stack.append(encode_num(1))
    else:
        stack.append(encode_num(0))
    return True

def op_add(stack):
    if len(stack) < 2:
        return False
    a = stack.pop()
    b = stack.pop()
    # 確保 a、b 是整數，如果是 bytes 就轉換
    if isinstance(a, bytes):
        a = int.from_bytes(a, 'big')
    if isinstance(b, bytes):
        b = int.from_bytes(b, 'big')
    stack.append(encode_num(a + b))
    return True

def op_mul(stack : list ):
    if len(stack) < 2:
        return False
    a = stack.pop()
    b = stack.pop()
    if isinstance(a, bytes):
        a = int.from_bytes(a, 'big')
    if isinstance(b, bytes):
        b = int.from_bytes(b, 'big')
    stack.append(encode_num(a * b))
    return True


# Complete OP_add
import hashlib

def op_checksig(stack : list , z):
    from ALL_Class.Bitcoin_S256Point import S256Point , Signature
    if len(stack) < 2:
        return False
    
    pubkey = stack.pop()
    signature = stack.pop()

    # 移除最後一個字節的 hash type（SIGHASH_ALL）
    signature = signature[:-1]
    
    point = S256Point.parse(sec_bin=pubkey)
    signature = Signature.parse(signature)

    try:
        if point.verify(z , signature):
            stack.append(encode_num(1))
        else:
            stack.append(encode_num(0))
    except (ValueError, SyntaxError):
        return False
    return True 

def op_verify(stack):
    """
    Verifies the top element of the stack.
    If the top element is not true, it raises an error.
    """
    if len(stack) < 1:
        return False
    top_element = stack.pop()
    if top_element != 1:
        raise ValueError("Verification failed: top element is not true")
    return True

def op_equalverify(stack):
    """
    Checks if the top two elements of the stack are equal.
    If they are, it removes them; if not, it raises an error.
    """
    if len(stack) < 2:
        return False
    a = stack.pop()
    b = stack.pop()
    if a != b:
        raise ValueError("Equal verification failed: elements are not equal")
    return True

def op_checkmultisig(stack : list , z):
    from ALL_Class.Bitcoin_S256Point import S256Point , Signature
    if len(stack) < 1:
        return False
    n = decode_num(stack.pop())
    if len(stack) < n + 1:
        return False
    sec_pubkeys = []
    for _ in range(n):
        sec_pubkeys.append(stack.pop())
    m = decode_num(stack.pop())
    if len(stack) < m + 1:
        return False
    der_signatures = []
    for _ in range(m):
        der_signatures.append(stack.pop()[:-1])  # Each DER signature is assumed to be signed with SIGHASH_ALL 
    stack.pop()  # Take care of the off-by-one error by consuming the only remaining element of the stack and not doing anything with the element
    
    try:
        for i in range(len(sec_pubkeys)):
            sec_pubkeys[i] = S256Point.parse(sec_bin=sec_pubkeys[i])

        for i in range(len(der_signatures)):
            der_signatures[i] = Signature.parse(der_signatures[i])
        
        pubkey_index = 0
        sig_index = 0
        
        # 因為不能用跳過的pubkey -> 使用 index 
        while sig_index < len(der_signatures) and pubkey_index < len(sec_pubkeys):
            if sec_pubkeys[pubkey_index].verify(z, der_signatures[sig_index]):  # if success 
                sig_index += 1  
            pubkey_index += 1 # No matter what happen 
        if sig_index == len(der_signatures):
            stack.append(encode_num(1))
        else:
            stack.append(encode_num(0))
        # The part that you need to code for this problem
    except (ValueError, SyntaxError):
        return False
    return True

OP_CODE_FUNCTIONS = {
    0 : op_0,
    81 : op_1,
    82 : op_2,
    86 : op_6, 
    105 : op_verify,
    135 : op_equal, 
    136 : op_equalverify,
    147 : op_add, 
    149 : op_mul, 
    118 : op_dup,
    166 : op_ripemd160,
    168 : op_sha256,
    169 : op_hash160,
    170 : op_hash256,
    172 : op_checksig,
    174 : op_checkmultisig
}

OP_CODE_NAMES = {
    0 : "op_0",
    81 : "op_1",
    82 : "op_2",
    86 : "op_6", 
    105 : "op_verify",
    135 : "op_equal", 
    136 : "op_equalverify",
    147 : "op_add", 
    149 : "op_mul", 
    118 : "op_dup",
    166 : "op_ripemd160",
    168 : "op_sha256",
    169 : "op_hash160",
    170 : "op_hash256",
    172 : "op_checksig",
    174 : "op_checkmultisig"
}

def encode_num(num):
    if num == 0 :
        return b''
    abs_num = abs(num)
    negative = num < 0 
    result = bytearray()
    while abs_num:
        result.append(abs_num & 0xff)
        abs_num >>= 8
    
    if result[-1] & 0x80:
        if negative:
            result.append(0x80)
        else:
            result.append(0x00)
    elif negative:
        result[-1] |= 0x80
    
    return bytes(result)

def decode_num(element):
    if element == b'':
        return 0 
    big_endian = element[::-1]
    if big_endian[0] & 0x80:
        negative = True
        result = big_endian[0] & 0x7f
    else:
        negative = False
        result = big_endian[0]
    
    for c in big_endian[1:]:
        result <<= 8
        result += c 
    
    if negative:
        return -result
    else:
        return result

if __name__ == '__main__':
    print(encode_num(-258)) ## b'\x02\x81'
    print(decode_num(b'\x02\x81')) ## -258
    print(encode_num(0))
    print(encode_num(1))
    print(encode_num(2))