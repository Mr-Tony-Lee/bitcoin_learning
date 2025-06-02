print("-------------------------- Problem 1 --------------------------\n")
def problem1():
    # 匯入所需模組
    from ALL_Class.Helper import hash256 , little_endian_to_int, decode_base58, p2pkh_script
    from ALL_Class.Bitcoin_S256Point import PrivateKey
    from ALL_Class.TxInput import TxIn
    from ALL_Class.TxOutput import TxOut
    from ALL_Class.Transaction import Tx
    # 用 hash256("Tony Lee secret") 的小端整數當作私鑰來源
    secret = little_endian_to_int(hash256(b'Tony Lee secret'))
    # 建立私鑰物件
    private_key = PrivateKey(secret)
    # 印出對應的 testnet 地址（可驗證這組私鑰對應的收款地址）
    print(private_key.point.address(testnet=True))
    """
        這筆交易的目的：
        我們要花掉以下這筆交易輸出中的比特幣：
        來源 TXID: 2ef0512ca5548b83b6bad06fe00871fc5704c210a5203610f39288e591d90f92
        輸出位置: 0
        收款地址: myzg7gYLQDYpiduGEzsVAsiyX8CPp2m7a1 (找零用)
        要匯出給：mqivhPWBNqd2Uk2gotT6AjACUeJMGYK6xz（目標地址）
    """
    # 上一筆交易的 txid（hex 轉為 bytes）
    prev_tx = bytes.fromhex("2ef0512ca5548b83b6bad06fe00871fc5704c210a5203610f39288e591d90f92")
    # 要花的是第 0 個 output
    prev_index = 0
    # SegWit 不用 scriptSig，legacy P2PKH 用空白佔位
    script_sig = None 
    # 預設 sequence（0xffffffff 表示不啟用 RBF）
    sequence = 0xffffffff
    # 收款地址（目標地址）
    target_address = "mqivhPWBNqd2Uk2gotT6AjACUeJMGYK6xz"
    # 要匯出的金額（BTC）
    target_amount = 0.0015722
    # 找零地址（把剩餘錢轉回）
    change_address = "myzg7gYLQDYpiduGEzsVAsiyX8CPp2m7a1"
    # 找零金額（BTC）
    change_amount = 0.000006
    # 建立交易輸入清單
    tx_ins = []
    tx_ins.append(TxIn(prev_tx, prev_index, script_sig, sequence))
    # 建立交易輸出清單
    tx_outs = []
    # === 建立目標地址的輸出 ===
    h160 = decode_base58(target_address)  # 解碼 base58 得到 hash160
    script_pubkey = p2pkh_script(h160)    # 建立 P2PKH 的鎖定腳本
    target_satoshis = int(target_amount * 100_000_000)  # 轉為 satoshi
    tx_outs.append(TxOut(target_satoshis, script_pubkey))  # 新增輸出
    # === 建立找零地址的輸出 ===
    h160 = decode_base58(change_address)
    script_pubkey = p2pkh_script(h160)
    change_satoshis = int(change_amount * 100_000_000)
    tx_outs.append(TxOut(change_satoshis, script_pubkey))
    # 建立整筆交易物件（版本 1，locktime = 0）
    tx_obj = Tx(1, tx_ins, tx_outs, 0, testnet=True)
    # 簽署第 0 筆輸入（就是唯一的那一筆）
    print(tx_obj.sign_input(0, private_key))  # True 表示簽章成功
    # 輸出序列化後的交易 hex 字串
    print(tx_obj.serialize().hex())
problem1()
print("\n-------------------------- Problem 1 --------------------------\n")

def Get_Test_z():
    from io import BytesIO
    from ALL_Class.Helper import encode_varint, hash256, int_to_little_endian
    from ALL_Class.Transaction import Tx
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
    return z 

print("-------------------------- Problem 2 --------------------------\n")
def problem2():
    def checkmultisig():
        from ALL_Class.Script import Script
        hex_sec = '03b287eaf122eea69030a0e9feed096bed8045c8b98bec453e1ffac7fbdbd4bb71' # the second sec public key
        hex_der = '3045022100da6bee3c93766232079a01639d07fa869598749729ae323eab8eef53577d611b02207bef15429dcadce2121ea07f233115c6f09034c0be68db99980b9a6c5e754022' # the DER-encoded value that appears second in the ScriptSig of the transaction
        
        sec = bytes.fromhex(hex_sec)
        der = bytes.fromhex(hex_der)
        
        script_pubkey = Script([sec , 0x51 , 0xae])
        script_sig = Script([0x00 , der+b'\x01' , 0x51])
        combined_script = script_sig + script_pubkey

        z = Get_Test_z()
        print(combined_script.evaluate(z))
    checkmultisig()
problem2()
print("\n-------------------------- Problem 2 --------------------------\n")



print("-------------------------- Problem 3 --------------------------\n")
SIGHASH_ALL = 1
def problem3():
    """
    You need to: 
        1.	Modify the transaction
        2.	Start with version
        3.	Add number of inputs
        4.	Modify the single TxIn to have the ScriptSig to be the RedeemScript
        5.	Add the number of outputs
        6.	Add each output serialization
        7.	Add the locktime
        8.	Add the SIGHASH_ALL
        9.	Hash256 the result
        10.	Interpret as a Big-Endian number
        11.	Parse the S256Point
        12.	Parse the Signature
        13.	Verify that the point, z and signature work
    """
    from io import BytesIO
    from ALL_Class.Bitcoin_S256Point import S256Point, Signature
    from ALL_Class.Helper import encode_varint, hash256, int_to_little_endian
    from ALL_Class.Script import Script
    from ALL_Class.Transaction import Tx
    hex_tx = '0100000001868278ed6ddfb6c1ed3ad5f8181eb0c7a385aa0836f01d5e4789e6bd304d87221a000000db00483045022100dc92655fe37036f47756db8102e0d7d5e28b3beb83a8fef4f5dc0559bddfb94e02205a36d4e4e6c7fcd16658c50783e00c341609977aed3ad00937bf4ee942a8993701483045022100da6bee3c93766232079a01639d07fa869598749729ae323eab8eef53577d611b02207bef15429dcadce2121ea07f233115c6f09034c0be68db99980b9a6c5e75402201475221022626e955ea6ea6d98850c994f9107b036b1334f18ca8830bfff1295d21cfdb702103b287eaf122eea69030a0e9feed096bed8045c8b98bec453e1ffac7fbdbd4bb7152aeffffffff04d3b11400000000001976a914904a49878c0adfc3aa05de7afad2cc15f483a56a88ac7f400900000000001976a914418327e3f3dda4cf5b9089325a4b95abdfa0334088ac722c0c00000000001976a914ba35042cfe9fc66fd35ac2224eebdafd1028ad2788acdc4ace020000000017a91474d691da1574e6b3c192ecfb52cc8984ee7b6c568700000000'
    hex_sec = '03b287eaf122eea69030a0e9feed096bed8045c8b98bec453e1ffac7fbdbd4bb71' # the second sec public key
    hex_der = '3045022100da6bee3c93766232079a01639d07fa869598749729ae323eab8eef53577d611b02207bef15429dcadce2121ea07f233115c6f09034c0be68db99980b9a6c5e754022' # the DER-encoded value that appears second in the ScriptSig of the transaction
    hex_redeem_script = '475221022626e955ea6ea6d98850c994f9107b036b1334f18ca8830bfff1295d21cfdb702103b287eaf122eea69030a0e9feed096bed8045c8b98bec453e1ffac7fbdbd4bb7152ae'
    sec = bytes.fromhex(hex_sec)
    der = bytes.fromhex(hex_der)
    redeem_script = Script.parse(BytesIO(bytes.fromhex(hex_redeem_script)))
    stream = BytesIO(bytes.fromhex(hex_tx))
    tx = Tx.parse(stream)
    
    # 2. Start with version
    modified_tx = int_to_little_endian(tx.version, 4) 

    # 3. Add number of inputs
    modified_tx += encode_varint(len(tx.tx_ins))
    
    for tx_in in tx.tx_ins:
        modified_tx += tx_in.prev_tx[::-1]
        modified_tx += int_to_little_endian(tx_in.prev_index, 4)

        # 4. Modify the single TxIn to have the ScriptSig to be the RedeemScript
        modified_tx += bytes.fromhex(hex_redeem_script)

        modified_tx += int_to_little_endian(tx_in.sequence , 4)
    
    # 5. Add the number of outputs
    modified_tx += encode_varint(len(tx.tx_outs))
    
    # 6. Add each output serialization
    for tx_out in tx.tx_outs:
        modified_tx += tx_out.serialize()
    # 7. Add the locktime
    modified_tx += int_to_little_endian(tx.locktime, 4)
    
    # 8. Add the SIGHASH_ALL
    modified_tx += int_to_little_endian(SIGHASH_ALL, 4)
    # 9. Hash256 the result
    h256 = hash256(modified_tx)
    # 10. Interpret as a Big-Endian number
    z = int.from_bytes(h256, 'big')
    # 11. Parse the S256Point
    point = S256Point.parse(sec)
    # 12. Parse the Signature
    sig = Signature.parse(der)
    # 13. Verify that the point, z and signature work
    print(point.verify(z,sig))
problem3()
print("\n-------------------------- Problem 3 --------------------------\n")