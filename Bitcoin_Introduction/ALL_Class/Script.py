from io import BytesIO
from ALL_Class.Helper import *
from ALL_Class.op import * 
from ALL_Class.Module import * 

LOGGER = logging.getLogger(__name__)

class Script:
    def __init__(self, cmds = False):
        self.cmds = cmds if cmds else []

    @classmethod
    def parse(cls, s):
        """Take a byte stream and parse the script at the start , return a Script object"""
        # 1. 讀取 length
        # length is an integer in 1 byte
        length = read_varint(s)
        # 2. 讀取 cmds
        # cmds is a list of bytes
        count = 0  
        cmds = []
        while count < length:
            current = s.read(1)
            count += 1 
            current_byte = current[0]
            if current_byte >= 1 and current_byte <= 75:
                n = current_byte
                cmds.append(s.read(n))
                count += n
            elif current_byte == 76:
                data_length = little_endian_to_int(s.read(1))
                cmds.append(s.read(data_length))
                count += data_length + 1
            elif current_byte == 77:
                data_length = little_endian_to_int(s.read(2))
                cmds.append(s.read(data_length))
                count += data_length + 2
            else:
                op_code = current_byte
                cmds.append(op_code)
        if count != length:
            raise SyntaxError('script length mismatch')
        # return an instance of Script 
        return cls(cmds)
    
    def raw_serialize(self):
        result = b''
        for cmd in self.cmds:
            if type(cmd) == int:
                result += int_to_little_endian(cmd, 1)
            else:
                length = len(cmd)
                if length <= 75:
                    result += int_to_little_endian(length, 1)
                elif length > 75 and length < 256:
                    result += int_to_little_endian(76, 1) + int_to_little_endian(length, 1)
                elif length >= 256 and length <= 520:
                    result += int_to_little_endian(77, 1) + int_to_little_endian(length, 2)
                else:
                    raise ValueError('script length too long')
                result += cmd
        return result
    def serialize(self):
        result = self.raw_serialize()
        total = len(result) 
        return encode_varint(total) + result
    
    def __add__(self,other):
        """Add two scripts together"""
        if type(other) == Script:
            return Script(self.cmds + other.cmds)
        else:
            raise TypeError('other must be a Script')
    def is_p2sh_script_pubkey(self):
        return len(self.cmds) == 3 and self.cmds[0] == 0xa9 and type(self.cmds[1]) == bytes and len(self.cmds[1]) == 20 and self.cmds[2] == 0x87
    def evaluate(self , z : list):
        cmds = self.cmds[:]
        stack = []
        altstack = []
        while len(cmds) > 0:
            cmd = cmds.pop(0)
            if type(cmd) == int:
                operation = OP_CODE_FUNCTIONS[cmd]
                # 1. 判斷是否為 op_code
                if cmd in (99,100):
                    if not operation(stack,cmds):
                        LOGGER.info("bad op: {}".format(OP_CODE_NAMES[cmd]))
                        return False
                elif cmd in (107,108):
                    if not operation(stack,altstack):
                        LOGGER.info("bad op: {}".format(OP_CODE_NAMES[cmd]))
                        return False
                elif cmd in (172,173,174,175):
                    if not operation(stack,z):
                        LOGGER.info("bad op: {}".format(OP_CODE_NAMES[cmd]))
                        return False
                else:
                    if not operation(stack):
                        LOGGER.info("bad op: {}".format(OP_CODE_NAMES[cmd]))
                        return False
            else:
                # 2. 判斷是否為資料
                stack.append(cmd)
                if self.is_p2sh_script_pubkey():
                    cmds.pop()
                    h160 = cmds.pop()
                    cmds.pop()
                    if not op_hash160(stack):
                        return False
                    stack.append(h160)
                    if not op_equal(stack):
                        return False
                    if not op_verify(stack):
                        LOGGER.info("bad p2sh h160")
                        return False
                    redeem_script = encode_varint(len(cmd)) + cmd
                    stream = BytesIO(redeem_script)
                    cmds.extend(Script.parse(stream).cmds)
        # 3. 判斷 stack 是否為空
        if len(stack) == 0:
            return False 
        if stack.pop == b'':
            return False
        return True
    
    def __repr__(self):
        result = []
        for cmd in self.cmds:
            if type(cmd) == int:
                if OP_CODE_NAMES.get(cmd):
                    name = OP_CODE_NAMES.get(cmd)
                else:
                    name = 'OP_[{}]'.format(cmd)
                result.append(name)
            else:
                result.append(cmd.hex())
        return ' '.join(result)


