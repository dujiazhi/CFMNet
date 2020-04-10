import torch
import torch.nn as nn

def resblock2_1(c,n_block,):
	attBlock = nn.ModuleList()
	attBlock.append(ResBlock_1(c))
	for i in range(n_block-1):
		attBlock.append(ResBlock_12(c))
	return attBlock


class RGBlock(nn.Module):    #residual guidence block
	def __init__(self,channel,n_block):
		super(RGBlock, self).__init__()
		layers = []
		layers.append(CFMResBlock(channel))
		for i in range(n_block-1):
			layers.append(FMResBlock(channel))
		self.rgblock=nn.Sequential(*layers)
	def forward(self,input):
		out=self.rgblock(input)
		return out

class ResBlock_1(nn.Module):
	def __init__(self, c):
		super(ResBlock_1, self).__init__()
		self.trans=nn.Sequential(
			nn.Conv2d(2*c, c, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(c),
			nn.ReLU(inplace=True))
		self.conv_block_stream1 = self.build_conv_block(c,c)
		self.conv_block_stream2 = self.build_conv_block(c,c)
		self.gamma=nn.Sequential(
			nn.Conv2d(c, c, kernel_size=3, padding=1),
			nn.Sigmoid())

	def build_conv_block(self, inc,outc):
		conv_block = []
		conv_block += [nn.Conv2d(inc, outc, kernel_size=3, padding=1, bias=False),
					   nn.BatchNorm2d(outc),
					   nn.ReLU(inplace=True)]
		conv_block += [nn.Conv2d(outc, outc, kernel_size=3, padding=1, bias=False),
					   nn.BatchNorm2d(outc),
					   nn.ReLU(inplace=True)]
		return nn.Sequential(*conv_block)


	def forward(self, x1, x2):
		# stream2 receive feedback from stream1
		org2=x2
		x2=torch.cat((x1,x2),1)
		x2=self.trans(x2)
		x1_out = self.conv_block_stream1(x1)
		x2_out = self.conv_block_stream2(x2)
		gamma=self.gamma(x2_out)

		x1_out = x1_out*gamma

		x1_out= x1 + x1_out # residual connection
		x2_out=org2+x2_out
		return x1_out, x2_out

class CFMResBlock(nn.Module):# conditional feature modulation
	def __init__(self, c):
		super(CFMResBlock, self).__init__()
		self.trans=nn.Sequential(
			nn.Conv2d(2*c, c, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(c),
			nn.ReLU(inplace=True))
		self.conv_feature = self.build_conv_block(c,c)
		self.conv_noisemap = self.build_conv_block(c,c)
		self.gamma=nn.Sequential(
			nn.Conv2d(c, c, kernel_size=3, padding=1),
			nn.Sigmoid())

	def build_conv_block(self, inc,outc):
		conv_block = []
		conv_block += [nn.Conv2d(inc, outc, kernel_size=3, padding=1, bias=False),
					   nn.BatchNorm2d(outc),
					   nn.ReLU(inplace=True)]
		conv_block += [nn.Conv2d(outc, outc, kernel_size=3, padding=1, bias=False),
					   nn.BatchNorm2d(outc),
					   nn.ReLU(inplace=True)]
		return nn.Sequential(*conv_block)


	def forward(self, input):
		(feature,noise_map)=input
		org_nm=noise_map
		noise_map=torch.cat((feature,noise_map),1)
		noise_map=self.trans(noise_map)
		f_out = self.conv_feature(feature)
		nm_out = self.conv_noisemap(noise_map)
		gamma=self.gamma(nm_out)
		f_out = f_out*gamma
		f_out= feature + f_out # residual connection
		nm_out=org_nm+nm_out
		return (f_out, nm_out)

class ResBlock_12(nn.Module):
	def __init__(self, c):
		super(ResBlock_12, self).__init__()
		self.conv_block_stream1 = self.build_conv_block(c,c)
		self.conv_block_stream2 = self.build_conv_block(c,c)

		self.gamma=nn.Sequential(nn.Conv2d(c, c, kernel_size=3, padding=1),
								 nn.Sigmoid())
	def build_conv_block(self, inc,outc):
		conv_block = []
		conv_block += [nn.Conv2d(inc, outc, kernel_size=3, padding=1, bias=False),
					   nn.BatchNorm2d(outc),
					   nn.ReLU(inplace=True)]
		conv_block += [nn.Conv2d(outc, outc, kernel_size=3, padding=1, bias=False),
					   nn.BatchNorm2d(outc),
					   nn.ReLU(inplace=True)]
		return nn.Sequential(*conv_block)

	def forward(self, x1, x2):
		x1_out = self.conv_block_stream1(x1)
		x2_out = self.conv_block_stream2(x2)
		gamma=self.gamma(x2_out)

		x1_out = x1_out*gamma
		x1_out= x1 + x1_out # residual connection
		x2_out=x2+x2_out
		return x1_out, x2_out

class FMResBlock(nn.Module):#feature modulation
	def __init__(self, c):
		super(FMResBlock, self).__init__()
		self.conv_feature = self.build_conv_block(c,c)
		self.conv_noisemap = self.build_conv_block(c,c)

		self.gamma=nn.Sequential(nn.Conv2d(c, c, kernel_size=3, padding=1),
								 nn.Sigmoid())
	def build_conv_block(self, inc,outc):
		conv_block = []
		conv_block += [nn.Conv2d(inc, outc, kernel_size=3, padding=1, bias=False),
					   nn.BatchNorm2d(outc),
					   nn.ReLU(inplace=True)]
		conv_block += [nn.Conv2d(outc, outc, kernel_size=3, padding=1, bias=False),
					   nn.BatchNorm2d(outc),
					   nn.ReLU(inplace=True)]
		return nn.Sequential(*conv_block)

	def forward(self, input):
		(feature, noise_map) = input
		f_out = self.conv_feature(feature)
		nm_out = self.conv_noisemap(noise_map)
		gamma=self.gamma(nm_out)

		f_out = f_out*gamma
		f_out= feature + f_out # residual connection
		nm_out=noise_map+nm_out
		return (f_out, nm_out)


class CFMNet(nn.Module):
	"""Custom U-Net architecture for Noise2Noise (see Appendix, Table 2)."""

	def __init__(self, in_channels, out_channels,nblock):
		"""Initializes U-Net."""
		super(CFMNet, self).__init__()
		n_block=nblock

		c = 64

		self.stream1_head= nn.Sequential(nn.Conv2d(in_channels, c, 3, stride=1, padding=1),nn.ReLU(inplace=True))
		self.stream2_head = nn.Sequential(nn.Conv2d(in_channels, c, 3, stride=1, padding=1),nn.ReLU(inplace=True))

		self.guided_down1=resblock2_1(c,n_block)

		self.stream1_down1 = nn.Sequential(nn.MaxPool2d(2),nn.Conv2d(c, 2*c, 3, stride=1, padding=1, bias=False),nn.BatchNorm2d(2*c),nn.ReLU(inplace=True))
		self.stream2_down1= nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(c, 2 * c, 3, stride=1, padding=1, bias=False),nn.BatchNorm2d(2*c),nn.ReLU(inplace=True))

		self.guided_down2 = resblock2_1(2*c,n_block)

		self.stream1_down2 = nn.Sequential(nn.MaxPool2d(2),nn.Conv2d(2*c, 4*c, 3, stride=1, padding=1, bias=False),nn.BatchNorm2d(4*c),nn.ReLU(inplace=True))
		self.stream2_down2= nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(2 * c, 4 * c, 3, stride=1, padding=1, bias=False),nn.BatchNorm2d(4*c),nn.ReLU(inplace=True))

		self.guided_down3 =resblock2_1(4*c,n_block)

		self.stream1_down3=nn.Sequential(nn.Conv2d(4*c, 4*c, 3, stride=1, padding=1, bias=False),nn.BatchNorm2d(4*c),nn.ReLU(inplace=True))
		self.stream2_down3=nn.Sequential(nn.Conv2d(4*c, 4*c, 3, stride=1, padding=1, bias=False),nn.BatchNorm2d(4*c),nn.ReLU(inplace=True))

		self.guided_up3= resblock2_1(4*c,n_block)

		self.stream1_up3 = nn.ConvTranspose2d(4*c, 2*c, 4, stride=2, padding=1, output_padding=0)
		self.stream2_up3 = nn.ConvTranspose2d(4 * c, 2 * c, 4, stride=2, padding=1, output_padding=0)

		self.stream1_up3_1 = nn.Sequential(nn.Conv2d(4*c, 2*c, 3, stride=1, padding=1, bias=False),nn.BatchNorm2d(2*c),nn.ReLU(inplace=True))
		self.stream2_up3_1 = nn.Sequential(nn.Conv2d(4*c, 2*c, 3, stride=1, padding=1, bias=False),nn.BatchNorm2d(2*c),nn.ReLU(inplace=True))

		self.guided_up2=resblock2_1(2*c,n_block)

		self.stream1_up2 =nn.ConvTranspose2d(2 * c, c, 4, stride=2, padding=1, output_padding=0)
		self.stream2_up2 =nn.ConvTranspose2d(2 * c, c, 4, stride=2, padding=1, output_padding=0)

		self.stream1_up2_1 = nn.Sequential(nn.Conv2d(2*c, c, 3, stride=1, padding=1, bias=False),nn.BatchNorm2d(c),nn.ReLU(inplace=True))
		self.stream2_up2_1 = nn.Sequential(nn.Conv2d(2*c, c, 3, stride=1, padding=1, bias=False),nn.BatchNorm2d(c),nn.ReLU(inplace=True))

		self.guided_up1=resblock2_1(c,n_block)

		self.tail = nn.Conv2d(c,out_channels, 3, stride=1, padding=1)

	def forward(self, x1,x2):
		"""Through encoder, then decoder by adding U-skip connections. """

		if x1.size()!= x2.size():
			x2=x2.view(-1,1,1,1).expand(x1.size())

		x1_down1=self.stream1_head (x1)
		x2_down1=self.stream2_head (x2)

		for model in self.guided_down1:
			x1_down1,x2_down1=model(x1_down1,x2_down1)

		concate1_1=x1_down1
		concate2_1=x2_down1
		x1_down2=self.stream1_down1 (x1_down1)#maxpool
		x2_down2 =self.stream2_down1 (x2_down1)

		for model in self.guided_down2:
			x1_down2, x2_down2 =model(x1_down2,x2_down2)

		concate1_2=x1_down2
		concate2_2 = x2_down2
		x1_down3 =self.stream1_down2(x1_down2)#maxpool
		x2_down3 =self.stream2_down2(x2_down2)

		for model in self.guided_down3:
			x1_down3, x2_down3 =model(x1_down3,x2_down3)

		x1_up3=self.stream1_down3 (x1_down3)
		x2_up3 =self.stream2_down3 (x2_down3)

		for model in self.guided_up3:
			x1_up3, x2_up3 =model(x1_up3,x2_up3)

		x1_up3=self.stream1_up3 (x1_up3)#convtranspose
		x2_up3 =self.stream1_up3 (x2_up3)

		x1_up2 =self.stream1_up3_1 (torch.cat((x1_up3,concate1_2),1))
		x2_up2 =self.stream2_up3_1 (torch.cat((x2_up3,concate2_2),1))

		for model in self.guided_up2:
			x1_up2,x2_up2=model(x1_up2,x2_up2)

		x1_up2 =self.stream1_up2  (x1_up2) #convtranspose
		x2_up2=self.stream2_up2  (x2_up2)

		x1_up1 =self.stream1_up2_1 (torch.cat((x1_up2,concate1_1),1))
		x2_up1 =self.stream2_up2_1  (torch.cat((x2_up2,concate2_1),1))

		for model in self.guided_up1:
			x1_up1, x2_up1 =model (x1_up1,x2_up1)

		x=self.tail (x1_up1)

		return x1-x


class CFMNet_(nn.Module):
	"""Custom U-Net architecture for Noise2Noise (see Appendix, Table 2)."""

	def __init__(self, in_channels, out_channels, nblock):
		"""Initializes U-Net."""
		super(CFMNet_, self).__init__()
		c = 64

		self.conv_feature_head= nn.Sequential(nn.Conv2d(in_channels, c, 3, stride=1, padding=1),nn.ReLU(inplace=True))
		self.conv_noisemap_head = nn.Sequential(nn.Conv2d(in_channels, c, 3, stride=1, padding=1),nn.ReLU(inplace=True))
		self.conv_guided_down1=RGBlock(c,nblock)

		self.conv_feature_down1 = nn.Sequential(nn.MaxPool2d(2),nn.Conv2d(c, 2*c, 3, stride=1, padding=1, bias=False),nn.BatchNorm2d(2*c),nn.ReLU(inplace=True))
		self.conv_noisemap_down1= nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(c, 2 * c, 3, stride=1, padding=1, bias=False),nn.BatchNorm2d(2*c),nn.ReLU(inplace=True))

		self.conv_guided_down2 = RGBlock(2*c,nblock)

		self.conv_feature_down2 = nn.Sequential(nn.MaxPool2d(2),nn.Conv2d(2*c, 4*c, 3, stride=1, padding=1, bias=False),nn.BatchNorm2d(4*c),nn.ReLU(inplace=True))
		self.conv_noisemap_down2= nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(2 * c, 4 * c, 3, stride=1, padding=1, bias=False),nn.BatchNorm2d(4*c),nn.ReLU(inplace=True))

		self.conv_guided_down3 =RGBlock(4*c,nblock)

		self.conv_feature_down3=nn.Sequential(nn.Conv2d(4*c, 4*c, 3, stride=1, padding=1, bias=False),nn.BatchNorm2d(4*c),nn.ReLU(inplace=True))
		self.conv_noisemap_down3=nn.Sequential(nn.Conv2d(4*c, 4*c, 3, stride=1, padding=1, bias=False),nn.BatchNorm2d(4*c),nn.ReLU(inplace=True))

		self.conv_guided_up3= RGBlock(4*c,nblock)

		self.conv_feature_up3_trans = nn.ConvTranspose2d(4*c, 2*c, 4, stride=2, padding=1, output_padding=0)
		self.conv_noisemap_up3_trans = nn.ConvTranspose2d(4 * c, 2 * c, 4, stride=2, padding=1, output_padding=0)

		self.conv_feature_up3 = nn.Sequential(nn.Conv2d(4*c, 2*c, 3, stride=1, padding=1, bias=False),nn.BatchNorm2d(2*c),nn.ReLU(inplace=True))
		self.conv_noisemap_up3 = nn.Sequential(nn.Conv2d(4*c, 2*c, 3, stride=1, padding=1, bias=False),nn.BatchNorm2d(2*c),nn.ReLU(inplace=True))

		self.conv_guided_up2=RGBlock(2*c,nblock)

		self.conv_feature_up2_trans =nn.ConvTranspose2d(2 * c, c, 4, stride=2, padding=1, output_padding=0)
		self.conv_noisemap_up2_trans =nn.ConvTranspose2d(2 * c, c, 4, stride=2, padding=1, output_padding=0)

		self.conv_feature_up2 = nn.Sequential(nn.Conv2d(2*c, c, 3, stride=1, padding=1, bias=False),nn.BatchNorm2d(c),nn.ReLU(inplace=True))
		self.conv_noisemap_up2 = nn.Sequential(nn.Conv2d(2*c, c, 3, stride=1, padding=1, bias=False),nn.BatchNorm2d(c),nn.ReLU(inplace=True))

		self.conv_guided_up1=RGBlock(c,nblock)

		self.conv_tail = nn.Conv2d(c,out_channels, 3, stride=1, padding=1)

	def forward(self, feature,noisemap):
		"""Through encoder, then decoder by adding U-skip connections. """
		if feature.size()!= noisemap.size():
			noisemap=noisemap.view(-1,1,1,1).expand(feature.size())

		feature_down1=self.conv_feature_head (feature)
		noisemap_down1=self.conv_noisemap_head (noisemap)
		(feature_down1,noisemap_down1)=self.conv_guided_down1((feature_down1,noisemap_down1))

		feature_down2=self.conv_feature_down1 (feature_down1)#maxpool
		noisemap_down2 =self.conv_noisemap_down1 (noisemap_down1)
		(feature_down2, noisemap_down2) = self.conv_guided_down2((feature_down2,noisemap_down2))

		feature_down3 =self.conv_feature_down2(feature_down2)#maxpool
		noisemap_down3 =self.conv_noisemap_down2(noisemap_down2)
		(feature_down3, noisemap_down3) =self.conv_guided_down3((feature_down3,noisemap_down3))

		feature_up3=self.conv_feature_down3 (feature_down3)
		noisemap_up3 =self.conv_noisemap_down3 (noisemap_down3)
		(feature_up3, noisemap_up3) =self.conv_guided_up3((feature_up3,noisemap_up3))

		feature_up3=self.conv_feature_up3_trans (feature_up3)#convtranspose
		noisemap_up3 =self.conv_feature_up3_trans (noisemap_up3)
		feature_up2 =self.conv_feature_up3(torch.cat((feature_up3,feature_down2),1))
		noisemap_up2 =self.conv_noisemap_up3(torch.cat((noisemap_up3,noisemap_down2),1))
		(feature_up2,noisemap_up2)=self.conv_guided_up2((feature_up2,noisemap_up2))

		feature_up2 =self.conv_feature_up2_trans(feature_up2) #convtranspose
		noisemap_up2=self.conv_noisemap_up2_trans(noisemap_up2)
		feature_up1 =self.conv_feature_up2(torch.cat((feature_up2,feature_down1),1))
		noisemap_up1 =self.conv_noisemap_up2(torch.cat((noisemap_up2,noisemap_down1),1))
		(feature_up1, noisemap_up1) =self.conv_guided_up1 ((feature_up1,noisemap_up1))
		x=self.conv_tail(feature_up1)
		return feature-x

#

net = CFMNet_(3, 3, 2).cuda()
input=torch.randn(3,3,8,8).cuda()
net(input,input)