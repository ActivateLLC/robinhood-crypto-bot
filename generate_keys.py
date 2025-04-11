#!/usr/bin/env python3
"""
Generate Ed25519 key pair for Robinhood Crypto API authentication.
"""

import nacl.signing
import base64

def generate_keys():
    # Generate an Ed25519 keypair
    private_key = nacl.signing.SigningKey.generate()
    public_key = private_key.verify_key

    # Convert keys to base64 strings
    private_key_base64 = base64.b64encode(private_key.encode()).decode()
    public_key_base64 = base64.b64encode(public_key.encode()).decode()

    # Print the keys in base64 format
    print("Private Key (Base64):")
    print(private_key_base64)
    print("\nPublic Key (Base64):")
    print(public_key_base64)
    
    print("\nIMPORTANT: Save these keys securely. You will need the public key to create")
    print("your API credentials in the Robinhood API Credentials Portal, and the private")
    print("key to authenticate your API requests. Never share your private key with anyone.")
    
    # Return the keys
    return private_key_base64, public_key_base64

if __name__ == "__main__":
    generate_keys()
