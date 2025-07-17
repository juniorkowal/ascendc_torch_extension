#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================

import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import sys, os

sys.path.append(os.getcwd())
import gather_custom

torch.npu.config.allow_internal_format = False


class TestCustomAdd(TestCase):

    # def test_gather_custom_ops(self):
    #     length = [8, 2048]
    #     x = torch.rand(length, device='cpu', dtype=torch.float16)
    #     y = torch.rand(length, device='cpu', dtype=torch.float16)

    #     x_npu = x.npu()
    #     y_npu = y.npu()
    #     output = gather_custom.run_gather_custom(x_npu, y_npu)
    #     cpuout = torch.add(x, y)

    #     self.assertRtolEqual(output, cpuout)
     def test_simple_gather_ops(self):

        input = torch.tensor([[1,2,3,4,1,2,3,4],[6,5,7,8,5,6,7,8]], dtype=torch.float32)
        index = torch.tensor([[0,4,0,1,1,1,2,2],[1,6,2,2,1,1,2,2]], dtype=torch.int64)
        dim=-1
        org=torch.gather(input,dim,index)

        index = index.to(torch.device('npu:0'))
        input = input.to(torch.device('npu:0'))

        my = gather_custom.run_gather_custom(input,dim,index)
        my = my.cpu()
        
        
        if not torch.equal(my,org):
            self.fail(f"Is not same \nMy= {my}\n Org = {org}\n")

    #  def test_attention(self):

    #     input=torch.rand(3,111,256,512, device='npu')
    #     index=torch.randint(low=1, high=250, size=[3,111,256,256],device='npu')
    #     dim=-1
    #     org=torch.gather(input,dim,index)
    #     index = index.npu()
    #     input = input.npu()

    #     my = gather_custom.run_gather_custom(input,dim,index)
    #     my = my.cpu()
        
    #     if not torch.equal(my,org):
    #         self.fail(f"Is not same \nMy= {my}\n Org = {org}\n")




if __name__ == "__main__":
    run_tests()
