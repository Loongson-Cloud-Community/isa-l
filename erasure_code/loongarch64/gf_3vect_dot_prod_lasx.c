/**********************************************************************
  Copyright(c) 2022 Loongson Tech All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions
  are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in
      the documentation and/or other materials provided with the
      distribution.
    * Neither the name of Intel Corporation nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**********************************************************************/

#include <lasxintrin.h>

void gf_3vect_dot_prod_lasx(int len, int vlen, unsigned char *gftbls,
			    unsigned char **src, unsigned char **dest)
{
	int i, j, l;
	int skip1 = vlen << 5;
	int skip2 = skip1 << 1;

	int block_count = len >> 5;
	int remain = len - (block_count << 5);
	int buf_offset = 0, col_offset = 0;

	__m256i block, high_idx, low_idx;
	__m256i high_elem, low_elem;
       	__m256i	high_val, low_val;
	__m256i res1, res2, res3;

	unsigned long sul, index, tmp;
	unsigned char *pos;
	unsigned char suc, sval;

	for (i = 0; i < block_count; ++i) {
		res1 = __lasx_xvldi(0);
		res2 = __lasx_xvldi(0);
		res3 = __lasx_xvldi(0);

		buf_offset = i << 5;
		for (j = 0; j < vlen; ++j) {
			/* read one block from one buf to update three rows */

			col_offset = j << 5;

			// fetch one block from one buf into a vector variable,
			// the block size is 32 bytes
			block = __lasx_xvldx(src[j], buf_offset);

			// get low indexes
			low_idx = __lasx_xvandi_b(block, 0x0F);

			// get high indexes: shift right 4 bits and add 0x10,
			// which can index {0x10, ..., 0x1F}
			high_idx = __lasx_xvori_b(__lasx_xvsrli_b(block, 4), 0x10);

			/* =================== for row 0 =================== */
			// fetch one element from gftbls, which is 32 bytes
			low_elem = __lasx_xvldx(gftbls, col_offset);
			high_elem = low_elem;

			// duplicate low part
			low_elem = __lasx_xvpermi_d(low_elem, 0x44);

			// duplicate high part
			high_elem = __lasx_xvpermi_d(high_elem, 0xEE);

			// pick by low_idx
			low_val = __lasx_xvshuf_b(low_elem, low_elem, low_idx);

			// pick by high_idx
			high_val = __lasx_xvshuf_b(high_elem, high_elem, high_idx);

			// accumulation on GF(2^8)
			res1 = __lasx_xvxor_v(res1, __lasx_xvxor_v(low_val, high_val));

			/* =================== for row 1 =================== */
			// fetch one element from gftbls, which is 32 bytes
			low_elem = __lasx_xvldx(gftbls, col_offset + skip1);
			high_elem = low_elem;

			// duplicate low part
			low_elem = __lasx_xvpermi_d(low_elem, 0x44);

			// duplicate high part
			high_elem = __lasx_xvpermi_d(high_elem, 0xEE);

			// pick by low_idx
			low_val = __lasx_xvshuf_b(low_elem, low_elem, low_idx);

			// pick by high_idx
			high_val = __lasx_xvshuf_b(high_elem, high_elem, high_idx);

			// accumulation on GF(2^8)
			res2 = __lasx_xvxor_v(res2, __lasx_xvxor_v(low_val, high_val));

			/* =================== for row 2 =================== */
			// fetch one element from gftbls, which is 32 bytes
			low_elem = __lasx_xvldx(gftbls, col_offset + skip2);
			high_elem = low_elem;

			// duplicate low part
			low_elem = __lasx_xvpermi_d(low_elem, 0x44);

			// duplicate high part
			high_elem = __lasx_xvpermi_d(high_elem, 0xEE);

			// pick by low_idx
			low_val = __lasx_xvshuf_b(low_elem, low_elem, low_idx);

			// pick by high_idx
			high_val = __lasx_xvshuf_b(high_elem, high_elem, high_idx);

			// accumulation on GF(2^8)
			res3 = __lasx_xvxor_v(res3, __lasx_xvxor_v(low_val, high_val));
		}

		__lasx_xvstx(res1, dest[0], buf_offset);
		__lasx_xvstx(res2, dest[1], buf_offset);
		__lasx_xvstx(res3, dest[2], buf_offset);
	}

	if (remain == 0)
		return;

	// handle remain bytes
	i = block_count << 5;
	while (remain >= 8) {
		for (l = 0; l < 3; ++l) {
			sul = 0;
			for (j = 0; j < vlen; ++j) {
				index = *(unsigned long *)&src[j][i];
				pos = &gftbls[(j << 5) + l * skip1];

				tmp = pos[index & 0xF] ^ pos[16 + ((index >> 4) & 0xF)];
				index = index >> 8;
				tmp |= (unsigned long)(pos[index & 0xF] ^ pos[16 + ((index >> 4) & 0xF)]) << 8;
				index = index >> 8;
				tmp |= (unsigned long)(pos[index & 0xF] ^ pos[16 + ((index >> 4) & 0xF)]) << 16;
				index = index >> 8;
				tmp |= (unsigned long)(pos[index & 0xF] ^ pos[16 + ((index >> 4) & 0xF)]) << 24;
				index = index >> 8;
				tmp |= (unsigned long)(pos[index & 0xF] ^ pos[16 + ((index >> 4) & 0xF)]) << 32;
				index = index >> 8;
				tmp |= (unsigned long)(pos[index & 0xF] ^ pos[16 + ((index >> 4) & 0xF)]) << 40;
				index = index >> 8;
				tmp |= (unsigned long)(pos[index & 0xF] ^ pos[16 + ((index >> 4) & 0xF)]) << 48;
				index = index >> 8;
				tmp |= (unsigned long)(pos[index & 0xF] ^ pos[16 + ((index >> 4) & 0xF)]) << 56;

				sul ^= tmp;
			}
			*(unsigned long *)&dest[l][i] = sul;
		}
		i += sizeof(unsigned long);
		remain -= sizeof(unsigned long);
	}

	if (remain == 0)
		return;

	buf_offset = i;
	for (l = 0; l < 3; l++) {
		for (i = buf_offset; i < len; i++) {
			suc = 0;
			for (j = 0; j < vlen; j++) {
				sval = src[j][i];
				pos = &gftbls[(j << 5) + l * skip1];
				suc ^= pos[sval & 0xF] ^ pos[16 + (sval >> 4)];
			}
			dest[l][i] = suc;
		}
	}
}
