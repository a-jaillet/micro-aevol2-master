// ***************************************************************************************************************
//
// Adrien Jaillet bitset
//
// ***************************************************************************************************************

#pragma once
#include <cmath>

#include <iostream>
#include <cstdint>

/**
 * Class that generates the mutation events for a given Organism
 */
struct cuBitSet {

    inline __device__ void init(uint new_size) {
        size = new_size;
        nb_bytes = ceil((float) size / 8.0f);
        bytes = (char *) malloc(nb_bytes);

        for (int i = 0; i < nb_bytes; i++)
        {
            bytes[i] = (char) 0b00000000;
        }
    }

    inline __device__
    void init_size(uint new_size) {
        size = new_size;
        nb_bytes = ceil((float) size / 8.0f);
    }

    inline __device__
    void free_it() {
        if (bytes != nullptr)
            free(bytes);
    }

    inline __device__
    void set(uint index) {
        if (index >= size)
        {
            printf("Error in Bitset: index > size\n");
        }
        uint blockid = index / 8;
        int num = index % 8;

        if (num == 0)
        {
            bytes[blockid] = bytes[blockid] ^ (char) 0x80;
        }
        if (num == 1)
        {
            bytes[blockid] = bytes[blockid] ^ (char) 0x40;
        }
        if (num == 2)
        {
            bytes[blockid] = bytes[blockid] ^ (char) 0x20;
        }
        if (num == 3)
        {
            bytes[blockid] = bytes[blockid] ^ (char) 0x10;
        }
        if (num == 4)
        {
            bytes[blockid] = bytes[blockid] ^ (char) 0x08;
        }
        if (num == 5)
        {
            bytes[blockid] = bytes[blockid] ^ (char) 0x04;
        }
        if (num == 6)
        {
            bytes[blockid] = bytes[blockid] ^ (char) 0x02;
        }
        if (num == 7)
        {
            bytes[blockid] = bytes[blockid] ^ (char) 0x01;
        }
    }
    
    inline __device__
    void set(uint index, int val) {
        if (index >= size)
        {
            printf("Error in Bitset: index > size\n");
        }
        if (val > 1)
        {
            printf("Error in Bitset: not valid val\n");
        }

        uint blockid = index / 8;
        int num = index % 8;

        if(val == 0)
        {
            if (num == 0)
            {
                bytes[blockid] = bytes[blockid] & (char) 0x7F;
            }
            if (num == 1)
            {
                bytes[blockid] = bytes[blockid] & (char) 0xBF;
            }
            if (num == 2)
            {
                bytes[blockid] = bytes[blockid] & (char) 0xDF;
            }
            if (num == 3)
            {
                bytes[blockid] = bytes[blockid] & (char) 0xEF;
            }
            if (num == 4)
            {
                bytes[blockid] = bytes[blockid] & (char) 0xF7;
            }
            if (num == 5)
            {
                bytes[blockid] = bytes[blockid] & (char) 0xFB;
            }
            if (num == 6)
            {
                bytes[blockid] = bytes[blockid] & (char) 0xFD;
            }
            if (num == 7)
            {
                bytes[blockid] = bytes[blockid] & (char) 0xFE;
            }
        }
        else
        {
            if (num == 0)
            {
                bytes[blockid] = bytes[blockid] | (char) 0x80;
            }
            if (num == 1)
            {
                bytes[blockid] = bytes[blockid] | (char) 0x40;
            }
            if (num == 2)
            {
                bytes[blockid] = bytes[blockid] |(char) 0x20;
            }
            if (num == 3)
            {
                bytes[blockid] = bytes[blockid] | (char) 0x10;
            }
            if (num == 4)
            {
                bytes[blockid] = bytes[blockid] | (char) 0x08;
            }
            if (num == 5)
            {
                bytes[blockid] = bytes[blockid] | (char) 0x04;
            }
            if (num == 6)
            {
                bytes[blockid] = bytes[blockid] | (char) 0x02;
            }
            if (num == 7)
            {
                bytes[blockid] = bytes[blockid] | (char) 0x01;
            }
        }
        
    }

    inline __device__
    int get(uint index) {
        if (index >= size)
        {
            printf("Error in Bitset: index > size\n");
        }

        uint blockid = index / 8;
        int j = index % 8;
        if (j == 0)
            return bytes[blockid] & 0x80 ? 1 : 0;
        if (j == 1)
            return bytes[blockid] & 0x40 ? 1 : 0;
        if (j == 2)
            return bytes[blockid] & 0x20 ? 1 : 0;
        if (j == 3)
            return bytes[blockid] & 0x10 ? 1 : 0;
        if (j == 4) 
            return bytes[blockid] & 0x08 ? 1 : 0;
        if (j == 5)
            return bytes[blockid] & 0x04 ? 1 : 0;
        if (j == 6)
            return bytes[blockid] & 0x02 ? 1 : 0;
        if (j == 7) 
            return bytes[blockid] & 0x01 ? 1 : 0;
        return 0;
    }
    
    inline __device__
    uint getSize()
    {
        return size;
    }

    inline __device__
    void print()
    {
        for (int i = 0; i < nb_bytes; i++)
        {
            for(int j = 0; j < 8; j++)
            {
                if (i * 8 + j < size)
                {
                    if (j == 0)
                        printf("%c", (bytes[i] & 0x80 ? '1' : '0'));
                    if (j == 1)
                        printf("%c", (bytes[i] & 0x40 ? '1' : '0'));
                    if (j == 2)
                        printf("%c", (bytes[i] & 0x20 ? '1' : '0'));
                    if (j == 3)
                        printf("%c", (bytes[i] & 0x10 ? '1' : '0'));
                    if (j == 4) 
                        printf("%c", (bytes[i] & 0x08 ? '1' : '0'));
                    if (j == 5)
                        printf("%c", (bytes[i] & 0x04 ? '1' : '0'));
                    if (j == 6)
                        printf("%c", (bytes[i] & 0x02 ? '1' : '0'));
                    if (j == 7) 
                        printf("%c", (bytes[i] & 0x01 ? '1' : '0'));
                }
            }
        }
        printf("\n");
    }

    inline __device__
    void getBits(char * return_val) {  
        uint idx = threadIdx.x;
        uint rr_width = blockDim.x;      
        
        for (uint position = idx; position < nb_bytes; position += rr_width) {    
            for(int j = 0; j < 8; j++)
            {
                if (position * 8 + j < size)
                {
                    if (j == 0)
                        return_val[position * 8 + j] = bytes[position] & 0x80 ? '1' : '0';
                    if (j == 1)
                        return_val[position * 8 + j] = bytes[position] & 0x40 ? '1' : '0';
                    if (j == 2)
                        return_val[position * 8 + j] = bytes[position] & 0x20 ? '1' : '0';
                    if (j == 3)
                        return_val[position * 8 + j] = bytes[position] & 0x10 ? '1' : '0';
                    if (j == 4) 
                        return_val[position * 8 + j] = bytes[position] & 0x08 ? '1' : '0';
                    if (j == 5)
                        return_val[position * 8 + j] = bytes[position] & 0x04 ? '1' : '0';
                    if (j == 6)
                        return_val[position * 8 + j] = bytes[position] & 0x02 ? '1' : '0';
                    if (j == 7) 
                        return_val[position * 8 + j] = bytes[position] & 0x01 ? '1' : '0';
                }
            }
        }
    }

    char * bytes = nullptr;
    uint size;
    uint nb_bytes;
    
};

