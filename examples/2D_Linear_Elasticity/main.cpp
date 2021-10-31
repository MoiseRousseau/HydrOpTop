/*MinimalFEM

Author: Stanislav Pidhorskyi (Podgorskiy)
stanislav@podgorskiy.com
stpidhorskyi@mix.wvu.edu

The source code available here: https://github.com/podgorskiy/MinimalFEM/

The MIT License (MIT)

Copyright (c) 2015 Stanislav Pidhorskyi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.*/

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>

struct Element
{
	void CalculateStiffnessMatrix(const Eigen::Matrix3f& D, std::vector<Eigen::Triplet<float> >& triplets);

	Eigen::Matrix<float, 3, 6> B;
	int nodesIds[3];
};

struct Constraint
{
	enum Type
	{
		UX = 1 << 0,
		UY = 1 << 1,
		UXY = UX | UY
	};
	int node;
	Type type;
};

int                      	nodesCount;
Eigen::VectorXf          	nodesX;
Eigen::VectorXf          	nodesY;
Eigen::VectorXf          	loads;
std::vector< Element >   	elements;
std::vector< Constraint >	constraints;

void Element::CalculateStiffnessMatrix(const Eigen::Matrix3f& D, std::vector<Eigen::Triplet<float> >& triplets)
{
	Eigen::Vector3f x, y;
	x << nodesX[nodesIds[0]], nodesX[nodesIds[1]], nodesX[nodesIds[2]];
	y << nodesY[nodesIds[0]], nodesY[nodesIds[1]], nodesY[nodesIds[2]];
	
	Eigen::Matrix3f C;
	C << Eigen::Vector3f(1.0f, 1.0f, 1.0f), x, y;
	
	Eigen::Matrix3f IC = C.inverse();

	for (int i = 0; i < 3; i++)
	{
		B(0, 2 * i + 0) = IC(1, i);
		B(0, 2 * i + 1) = 0.0f;
		B(1, 2 * i + 0) = 0.0f;
		B(1, 2 * i + 1) = IC(2, i);
		B(2, 2 * i + 0) = IC(2, i);
		B(2, 2 * i + 1) = IC(1, i);
	}
	Eigen::Matrix<float, 6, 6> K = B.transpose() * D * B * std::abs(C.determinant()) / 2.0f;

	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			Eigen::Triplet<float> trplt11(2 * nodesIds[i] + 0, 2 * nodesIds[j] + 0, K(2 * i + 0, 2 * j + 0));
			Eigen::Triplet<float> trplt12(2 * nodesIds[i] + 0, 2 * nodesIds[j] + 1, K(2 * i + 0, 2 * j + 1));
			Eigen::Triplet<float> trplt21(2 * nodesIds[i] + 1, 2 * nodesIds[j] + 0, K(2 * i + 1, 2 * j + 0));
			Eigen::Triplet<float> trplt22(2 * nodesIds[i] + 1, 2 * nodesIds[j] + 1, K(2 * i + 1, 2 * j + 1));

			triplets.push_back(trplt11);
			triplets.push_back(trplt12);
			triplets.push_back(trplt21);
			triplets.push_back(trplt22);
		}
	}
}

void SetConstraints(Eigen::SparseMatrix<float>::InnerIterator& it, int index)
{
	if (it.row() == index || it.col() == index)
	{
		it.valueRef() = it.row() == it.col() ? 1.0f : 0.0f;
	}
}

void ApplyConstraints(Eigen::SparseMatrix<float>& K, const std::vector<Constraint>& constraints)
{
	std::vector<int> indicesToConstraint;

	for (std::vector<Constraint>::const_iterator it = constraints.begin(); it != constraints.end(); ++it)
	{
		if (it->type & Constraint::UX)
		{
			indicesToConstraint.push_back(2 * it->node + 0);
		}
		if (it->type & Constraint::UY)
		{
			indicesToConstraint.push_back(2 * it->node + 1);
		}
	}

	for (int k = 0; k < K.outerSize(); ++k)
	{
		for (Eigen::SparseMatrix<float>::InnerIterator it(K, k); it; ++it)
		{
			for (std::vector<int>::iterator idit = indicesToConstraint.begin(); idit != indicesToConstraint.end(); ++idit)
			{
				SetConstraints(it, *idit);
			}
		}
	}
}

int main(int argc, char *argv[])
{
	if ( argc != 2)
    {
        std::cout<<"usage: "<< argv[0] <<" <prefix> \n";
        return 1;
    }
	
	std::string prefix(argv[1]);
	std::ifstream meshfile(prefix+".mesh");
	if (meshfile.is_open() == false)
	{
	    std::cerr << "Error opening mesh file: " << prefix+".mesh" << std::endl;
	    return 1;
	}
	std::ifstream bcfile(prefix+".bcs");
	if (bcfile.is_open() == false)
	{
	    std::cerr << "Error opening boundary conditions file: " << prefix+".bcs" << std::endl;
	    return 1;
	}
	std::ifstream matpropfiles(prefix+".matprops");
	if (matpropfiles.is_open() == false)
	{
	    std::cerr << "Error opening material properties file: " << prefix+".matprops" << std::endl;
	    return 1;
	}
	
	
	// READ MESH //
	meshfile >> nodesCount;
	nodesX.resize(nodesCount);
	nodesY.resize(nodesCount);

	for (int i = 0; i < nodesCount; ++i)
	{
		meshfile >> nodesX[i] >> nodesY[i];
	}

	int elementCount;
	meshfile >> elementCount;

	for (int i = 0; i < elementCount; ++i)
	{
		Element element;
		meshfile >> element.nodesIds[0] >> element.nodesIds[1] >> element.nodesIds[2];
		elements.push_back(element);
	}
	

	// MAT PROPERTIES //
	//first line, poisson ratio
	float poissonRatio;
	matpropfiles >> poissonRatio;
	
	//young modulus
	std::vector<float> youngModulus;
	youngModulus.resize(elementCount);
    for (int i = 0; i < elementCount; ++i)
    {
	    matpropfiles >> youngModulus[i];
    }
	
	Eigen::Matrix3f D_;
	D_ <<
		1.0f,        	poissonRatio,	0.0f,
		poissonRatio,	1.0,         	0.0f,
		0.0f,        	0.0f,        	(1.0f - poissonRatio) / 2.0f;
	D_ *= 1 / (1.0f - pow(poissonRatio, 2.0f));


	// CONSTRAINTS AND LOAD//
	// Constraints
	int constraintCount;
	bcfile >> constraintCount;

	for (int i = 0; i < constraintCount; ++i)
	{
		Constraint constraint;
		int type;
		bcfile >> constraint.node >> type;
		constraint.type = static_cast<Constraint::Type>(type);
		constraints.push_back(constraint);
	}

    //loads
	loads.resize(2 * nodesCount);
	loads.setZero();

	int loadsCount;
	bcfile >> loadsCount;

	for (int i = 0; i < loadsCount; ++i)
	{
		int node;
		float x, y;
		bcfile >> node >> x >> y;
		loads[2 * node + 0] = x;
		loads[2 * node + 1] = y;
	}
	
	
	// PROBLEM ASSEMBLY
	std::vector<Eigen::Triplet<float> > triplets, Dtriplets;
    for (int i=0; i < elementCount; ++i) 
    {
        elements[i].CalculateStiffnessMatrix(D_*youngModulus[i], triplets);
        elements[i].CalculateStiffnessMatrix(D_, Dtriplets); //derivative
    }

	Eigen::SparseMatrix<float> globalK(2 * nodesCount, 2 * nodesCount);
	Eigen::SparseMatrix<float> DglobalK(2 * nodesCount, 2 * nodesCount);
	globalK.setFromTriplets(triplets.begin(), triplets.end()); //TODO: add values ?
	DglobalK.setFromTriplets(Dtriplets.begin(), Dtriplets.end());

	ApplyConstraints(globalK, constraints);
	
	// SOLVE
	Eigen::SimplicialLDLT<Eigen::SparseMatrix<float> > solver(globalK);
	Eigen::VectorXf displacements = solver.solve(loads);
	
	// OUTPUT
    //displacements
	std::ofstream outdisplacements(prefix+".displacements");
	for (int i=0; i < nodesCount; ++i) 
	{
	    outdisplacements << displacements[2*i] << " "; //x displacement
	    outdisplacements << displacements[2*i+1] << std::endl; //y displacement
	}
	
	outdisplacements.close();

    //von mises stress
	std::ofstream outstress(prefix+".stress");
    int count = 0;
	for (std::vector<Element>::iterator it = elements.begin(); it != elements.end(); ++it)
	{
		Eigen::Matrix<float, 6, 1> delta;
		delta << displacements.segment<2>(2 * it->nodesIds[0]),
		         displacements.segment<2>(2 * it->nodesIds[1]),
		         displacements.segment<2>(2 * it->nodesIds[2]);
		         
		Eigen::Vector3f sigma = D_ * youngModulus[count] * it->B * delta;
		
		float sigma_mises = sqrt(sigma[0] * sigma[0] - sigma[0] * sigma[1] + sigma[1] * sigma[1] + 3.0f * sigma[2] * sigma[2]);

		outstress << sigma_mises << std::endl;
		count++;
	}
	
	//sensitivity
	Eigen::saveMarket(globalK, prefix+"_jacobian.mtx");
	Eigen::saveMarket(DglobalK, prefix+"_sensitivity.mtx");
	
	return 0;
}
