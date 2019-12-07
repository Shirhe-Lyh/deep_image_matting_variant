# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 13:41:23 2019

@author: lijingxiong
"""

import torch

import conv_square


class ConvTest(torch.nn.Module):
    """A mini networt to test Conv2_5d in forward and backword computation."""
    
    def __init__(self, num_classes=2, a=None, b=None, c=None):
        super(ConvTest, self).__init__()
        
        self._head_conv = conv_square.ConvSquare(in_channels=3, 
                                                 out_channels=32,
                                                 kernel_size=5, 
                                                 padding=2, 
                                                 bias=False,
                                                 a=a, b=b, c=c)
        self._pred_conv = torch.nn.Conv2d(in_channels=32,
                                          out_channels=num_classes,
                                          kernel_size=3,
                                          padding=1,
                                          bias=False)
        self._batch_norm = torch.nn.BatchNorm2d(num_features=num_classes,
                                                momentum=0.995)
        
    def forward(self, x, z):
        x = self._head_conv(x, z)
        x = self._pred_conv(x)
        x = self._batch_norm(x)
        return x
    
    
if __name__ == '__main__':
    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    print(''.join(['-'] * 30))
    print('For parameters a, b, c')
    print(''.join(['-'] * 30))
    
    model = ConvTest().to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    
    num_steps = 100
    for i in range(num_steps):
        images = torch.rand((2, 3, 64, 64)).to(device)
        alpha = torch.rand((2, 1, 64, 64)).to(device)
        labels = torch.LongTensor(
            torch.full((2, 64, 64), 0, dtype=torch.int64)).to(device)
        
        # Forward pass
        outputs = model(images, alpha)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print('Step: {}/{}, Loss: {:.4f}'.format(i+1, num_steps, loss.item()))
        
    print(''.join(['-'] * 30))
    print('For constants a, b, c')
    print(''.join(['-'] * 30))
        
    model = ConvTest(a=-4, b=4, c=1).to(device)
        
    num_steps = 100
    for i in range(num_steps):
        images = torch.rand((2, 3, 64, 64)).to(device)
        alpha = torch.rand((2, 1, 64, 64)).to(device)
        labels = torch.LongTensor(
            torch.full((2, 64, 64), 0, dtype=torch.int64)).to(device)
        
        # Forward pass
        outputs = model(images, alpha)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print('Step: {}/{}, Loss: {:.4f}'.format(i+1, num_steps, loss.item()))

