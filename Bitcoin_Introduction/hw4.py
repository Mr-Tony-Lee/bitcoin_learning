print("-------------------------- Problem 1 --------------------------\n")
def problem1():
    pass
problem1()
print("\n-------------------------- Problem 1 --------------------------\n")


print("-------------------------- Problem 2 --------------------------\n")
def problem2():
    def checkmultisig():
        pass
    
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




