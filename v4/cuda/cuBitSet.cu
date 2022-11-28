__device__ void init(uint new_size) {
        size = new_size;
        nb_bytes = ceil((float) size / 8.0f);
        bytes = (char *) malloc(nb_bytes);

        for (int i = 0; i < nb_bytes; i++)
        {
            bytes[i] = (char) 0b00000000;
        }
    }