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

#include "erasure_code.h"
#include "ec_loongarch64.h"

void gf_vect_dot_prod(int len, int vlen, unsigned char *v,
		      unsigned char **src, unsigned char *dest)
{
	gf_vect_dot_prod_lasx(len, vlen, v, src, dest);
}

void gf_vect_mad(int len, int vec, int vec_i,
		 unsigned char *v, unsigned char *src, unsigned char *dest)
{
	gf_vect_mad_lasx(len, vec, vec_i, v, src, dest);
}

void ec_encode_data(int len, int srcs, int dests, unsigned char *v,
		    unsigned char **src, unsigned char **dest)
{
	ec_encode_data_lasx(len, srcs, dests, v, src, dest);
}

void ec_encode_data_update(int len, int k, int rows, int vec_i, unsigned char *v,
			   unsigned char *data, unsigned char **dest)
{
	ec_encode_data_update_lasx(len, k, rows, vec_i, v, data, dest);
}

int gf_vect_mul(int len, unsigned char *a, void *src, void *dest)
{
	gf_vect_mul_lasx(len, a, (unsigned char *)src, (unsigned char *)dest);
	return 0;
}
