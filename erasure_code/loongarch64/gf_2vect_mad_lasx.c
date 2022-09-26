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

void gf_2vect_mad_lasx(int len, int vec, int vec_i, unsigned char *gftbls,
		       unsigned char *src, unsigned char **dest)
{
	int i, l;
	int skip1 = vec << 5;

	int block_count = len >> 5;
	int remain = len - (block_count << 5);
	int buf_offset = 0, col_offset = vec_i << 5;

	__m256i block, high_idx, low_idx;
	__m256i high_elem, low_elem;
       	__m256i	high_val, low_val;
	__m256i res;

	unsigned long index, tmp, tmp1;
	unsigned char *pos;
	unsigned char sval;

	for (i = 0; i < block_count; ++i) {
		/* read one block from src to update two rows */

		buf_offset = i << 5;
		// fetch one block from one buf into a vector variable,
		// the block size is 32 bytes
		block = __lasx_xvldx(src, buf_offset);

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

		// load old data
		res = __lasx_xvldx(dest[0], buf_offset);

		// accumulation on GF(2^8)
		res = __lasx_xvxor_v(res, __lasx_xvxor_v(low_val, high_val));

		// store new data
		__lasx_xvstx(res, dest[0], buf_offset);

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

		// load old data
		res = __lasx_xvldx(dest[1], buf_offset);

		// accumulation on GF(2^8)
		res = __lasx_xvxor_v(res, __lasx_xvxor_v(low_val, high_val));

		// store new data
		__lasx_xvstx(res, dest[1], buf_offset);
	}

	if (remain == 0)
		return;

	// handle remain bytes
	i = block_count << 5;
	while (remain >= 8) {
		tmp1 = *(unsigned long *)&src[i];
		for (l = 0; l < 2; ++l) {
			index = tmp1;
			pos = &gftbls[col_offset + l * skip1];

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

			*(unsigned long *)&dest[l][i] ^= tmp;
		}
		i += sizeof(unsigned long);
		remain -= sizeof(unsigned long);
	}

	if (remain == 0)
		return;

	buf_offset = i;
	for (l = 0; l < 2; l++) {
		for (i = buf_offset; i < len; i++) {
			sval = src[i];
			pos = &gftbls[col_offset + l * skip1];
			dest[l][i] ^= pos[sval & 0xF] ^ pos[16 + (sval >> 4)];
		}
	}
}
