from ALL_Class.Transaction import Tx
from ALL_Class.Script import Script
from io import BytesIO



print("-------------------------- Problem 1 --------------------------\n")
def problem1():
    hex_transaction = "010000000117e18a4a4a0af876b1b0a4764ee77c74106e07667dd94c4d61271f3d356cbf62000000006b4830450221009e661e94622a66f6c65f270d859828360c825ee755d675c9cbb2214685ba08fc022005aa4abaf21a84519f0c8ff40c633a0e4a624c639d25c0ea908d0d5e463749a80121036ddc934a5fbd5222ead406a4334462aaa62f83d0b02255c0a582f9038a17bbfdffffffff02cc162c00000000001976a914051b07716871833694a762ad15565b86da46622488ac16ae0e00000000001976a914c03ee4258550c77bcf61829c7cb636cd521ebfc588ac00000000"
    tx_bytes = bytes.fromhex(hex_transaction)    
    stream = BytesIO(tx_bytes)
    tx = Tx.parse(stream)

    print("The ScriptSig from the first input :", tx.tx_ins[0].script_sig )
    print("ScriptPubKey from the first output :", tx.tx_outs[0].script_pubkey )
    print("The amount of the second output :", tx.tx_outs[1].amount)

problem1()
print("\n-------------------------- Problem 1 --------------------------\n")


print("-------------------------- Problem 2 --------------------------\n")
def problem2():
    def check_checksig():
        from ALL_Class.Script import Script
        from ALL_Class.Bitcoin_S256Point import PrivateKey
        scriptpubkey = '43410411db93e1dcdb8a016b49840f8c53bc1eb68a382e97b1482ecad7b148a6909a5cb2e0eaddfb84ccf9744464f82e160bfa9b8b64f9d4c03f999b8643f656b412a3ac'
        scriptsig = '4847304402204e45e16932b8af514961a1d3a1a25fdf3f4f7732e9d624c6c61548ab5fb8cd410220181522ec8eca07de4860a4acdd12909d831cc56cbbac4622082221a8768d1d0901'
        script_pubkey = Script().parse(BytesIO(bytes.fromhex(scriptpubkey)))
        script_sig = Script().parse(BytesIO(bytes.fromhex(scriptsig)))
        combined_script = script_sig + script_pubkey
        z = 0x7c076ff316692a3d7ebd292d4f6c744b3c48f5f05d39de12b4d4c1d1710f20ae
        print(combined_script.evaluate(z))
    check_checksig()
problem2()
print("\n-------------------------- Problem 2 --------------------------\n")

print("-------------------------- Problem 3 --------------------------\n")
def problem3():
    # 1. OP_DUP 重複 x，stack: x x
    # 2. OP_DUP 再重複頂端，stack: x x x
    # 3. OP_MUL 將頂兩元素相乘，stack: x x*x
    # 4. OP_ADD 將頂兩元素相加，stack: (x) + (x*x) = x + x^2
    # 5. OP_6 是常數6
    # 6. OP_EQUAL 比較前兩個元素是否相等

    # 所以 x + x^2 == 6，解這方程式:
    # x^2 + x - 6 = 0
    # 解得 x = 2 或 x = -3（只要輸入2即可）
    script_sig = Script([b'\x02'])  # push 2
    script_pubkey = Script([0x76, 0x76, 0x95, 0x93, 0x56, 0x87]) 
    combined_script = Script(script_sig.cmds + script_pubkey.cmds)
    result = Script.evaluate(combined_script, z=0)  # z=0 因為這不是簽名驗證腳本，z不影響
    print("Evaluate result:", result)  # True表示解鎖成功
problem3()
print("\n-------------------------- Problem 3 --------------------------\n")

