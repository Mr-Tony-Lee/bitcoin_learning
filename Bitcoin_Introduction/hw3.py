print("-------------------------- Problem 1 --------------------------\n")
def problem1():
    from ALL_Class.Transaction import Tx
    from io import BytesIO
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
        from io import BytesIO
        from ALL_Class.Bitcoin_S256Point import S256Point, Signature
        from ALL_Class.Helper import encode_varint, hash256, int_to_little_endian
        from ALL_Class.Script import Script
        from ALL_Class.Transaction import Tx
        hex_sec = '03b287eaf122eea69030a0e9feed096bed8045c8b98bec453e1ffac7fbdbd4bb71' # the second sec public key
        hex_der = '3045022100da6bee3c93766232079a01639d07fa869598749729ae323eab8eef53577d611b02207bef15429dcadce2121ea07f233115c6f09034c0be68db99980b9a6c5e754022' # the DER-encoded value that appears second in the ScriptSig of the transaction
        # 要加上 sighash_all 在 checksig 才會被通過
        sec = bytes.fromhex(hex_sec)
        der = bytes.fromhex(hex_der)
        hex_tx = '0100000001868278ed6ddfb6c1ed3ad5f8181eb0c7a385aa0836f01d5e4789e6bd304d87221a000000db00483045022100dc92655fe37036f47756db8102e0d7d5e28b3beb83a8fef4f5dc0559bddfb94e02205a36d4e4e6c7fcd16658c50783e00c341609977aed3ad00937bf4ee942a8993701483045022100da6bee3c93766232079a01639d07fa869598749729ae323eab8eef53577d611b02207bef15429dcadce2121ea07f233115c6f09034c0be68db99980b9a6c5e75402201475221022626e955ea6ea6d98850c994f9107b036b1334f18ca8830bfff1295d21cfdb702103b287eaf122eea69030a0e9feed096bed8045c8b98bec453e1ffac7fbdbd4bb7152aeffffffff04d3b11400000000001976a914904a49878c0adfc3aa05de7afad2cc15f483a56a88ac7f400900000000001976a914418327e3f3dda4cf5b9089325a4b95abdfa0334088ac722c0c00000000001976a914ba35042cfe9fc66fd35ac2224eebdafd1028ad2788acdc4ace020000000017a91474d691da1574e6b3c192ecfb52cc8984ee7b6c568700000000'
        hex_redeem_script = '475221022626e955ea6ea6d98850c994f9107b036b1334f18ca8830bfff1295d21cfdb702103b287eaf122eea69030a0e9feed096bed8045c8b98bec453e1ffac7fbdbd4bb7152ae'
        stream = BytesIO(bytes.fromhex(hex_tx))
        tx = Tx.parse(stream)
        modified_tx = int_to_little_endian(tx.version, 4) 
        modified_tx += encode_varint(len(tx.tx_ins))
        for tx_in in tx.tx_ins:
            modified_tx += tx_in.prev_tx[::-1]
            modified_tx += int_to_little_endian(tx_in.prev_index, 4)
            modified_tx += bytes.fromhex(hex_redeem_script)
            modified_tx += int_to_little_endian(tx_in.sequence , 4)
        modified_tx += encode_varint(len(tx.tx_outs))
        for tx_out in tx.tx_outs:
            modified_tx += tx_out.serialize()
        modified_tx += int_to_little_endian(tx.locktime, 4)
        modified_tx += int_to_little_endian(1, 4)
        h256 = hash256(modified_tx)

        z = int.from_bytes(h256, 'big')        
        script_pubkey = Script([sec , 0xac])
        script_sig = Script([der + b'\x01'])
        combined_script = script_sig + script_pubkey
        
        print(combined_script.evaluate(z))
    check_checksig()
problem2()
print("\n-------------------------- Problem 2 --------------------------\n")

print("-------------------------- Problem 3 --------------------------\n")
def problem3():
    from ALL_Class.Script import Script
    # 1. OP_DUP 重複 x，stack: x x
    # 2. OP_DUP 再重複頂端，stack: x x x
    # 3. OP_MUL 將頂兩元素相乘，stack: x x*x
    # 4. OP_ADD 將頂兩元素相加，stack: (x) + (x*x) = x + x^2
    # 5. OP_6 是常數6
    # 6. OP_EQUAL 比較前兩個元素是否相等

    # 所以 x + x^2 == 6，解這方程式:
    # x^2 + x - 6 = 0
    # 解得 x = 2 或 x = -3（只要輸入2即可）
    script_sig = Script([0x52])  # push 2
    script_pubkey = Script([0x76, 0x76, 0x95, 0x93, 0x56, 0x87]) 
    combined_script = Script(script_sig.cmds + script_pubkey.cmds)
    result = Script.evaluate(combined_script, z=0)  # z=0 因為這不是簽名驗證腳本，z不影響
    print("Evaluate result:", result)  # True表示解鎖成功
problem3()
print("\n-------------------------- Problem 3 --------------------------\n")

