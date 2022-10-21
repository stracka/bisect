{

  using namespace RooFit;

  int ntoys = 1000;

  /* ----- parameters for the simulation ----- */ 

  double asym=0.;
  int nruns = 2;
  double NtrapPerRun=50.;
  double NcosmPerRun=10.;

  double Ngen = nruns * NtrapPerRun; 
  double cosm = nruns * NcosmPerRun;
  
  double Mbal = 0.4;
  double Loct = 0.3;
  double trap = 1-Loct - Mbal;

  double geff[2] = {1.,1.}; // overall efficiencies for the two classes 
  double eff[2] = {geff[0],geff[1]};

  double hconf[2] = {0.8, 0.8};    // diagonal elements of the confusion matrix
  double cdistDn = 0.5;   // probability for a cosmic to appear in the "down" region

  
  /* ----- parameters for the fit ----- */
  
  const int nbins = 2;
  double zmin = -1; 
  double zmax = 1; 

  TH1D* hbarUp = new TH1D("histUp","histUp",nbins,zmin,zmax); 
  TH1D* hbarDn = new TH1D("histDn","histDn",nbins,zmin,zmax); 
  TH1D* hcosm  = new TH1D("hcosm" ,"hcosm" ,nbins,zmin,zmax); 

  hbarDn->SetBinContent(0+1,hconf[1]);
  hbarDn->SetBinContent(1+1,1-hconf[1]);

  hbarUp->SetBinContent(0+1,1-hconf[0]);
  hbarUp->SetBinContent(1+1,hconf[0]);
  
  hcosm->SetBinContent(0+1,cdistDn);
  hcosm->SetBinContent(1+1,1-cdistDn);

  RooRealVar z("z","",zmin,zmax);
  z.setBins(nbins);
  RooRealVar effUp("effUp","",eff[0]);
  RooRealVar effDn("effDn","",eff[1]);

  RooDataHist* rdhbarUp = new RooDataHist("rdhistUp","rdhistUp",RooArgList(z),hbarUp); 
  RooDataHist* rdhbarDn = new RooDataHist("rdhistDn","rdhistDn",RooArgList(z),hbarDn); 
  RooDataHist* rdhcosm  = new RooDataHist("rdhcosm" ,"rdhcosm" ,RooArgList(z),hcosm ); 
  
  RooHistPdf pbarUp("pbarUp","pbarUp",RooArgSet(z),*rdhbarUp,0);
  RooHistPdf pbarDn("pbarDn","pbarDn",RooArgSet(z),*rdhbarDn,0);
  RooHistPdf pcosm( "pcosm" ,"pcosm" ,RooArgSet(z),*rdhcosm ,0);

  RooPlot *zframe = z.frame(Title("extended ML fit example"));


  /* ----- poor man's bookkeeping ----- */
  
  vector<double> gasym;
  vector<double> par1;
  vector<double> err1;
  vector<double> par2;
  vector<double> err2;

  /* ----- repeat for a few values of the asymmetry ----- */
  
  for (int ia=0; ia<10+1; ia++){

    asym = -1. + ia*2./10;  
  
    TH1D hpull("hpull","hpull",50,-4,4);

    /* ----- perform a number of toys for each asymmetry ----- */ 

    for (int ie=0; ie<ntoys; ie++){

      /* ..... begin simulation ..... */
      
      // container for the generated data
      TH1D hdata("hdata","hdata",nbins,zmin,zmax); 

      // poisson to generate total number of Hbar in the trap 
      int muNgen = gRandom->Poisson(Ngen);

      for (int i=0; i<muNgen; i++){

	// split in ramp phases and keep only Mbal fraction
	if ( gRandom->Uniform(0,1) > Mbal ) continue; 

	// randomly split in up and down classes, according to the asymmetry
	double upProb =  (1.+asym)*0.5;  
	int isDn = ( gRandom->Uniform(0,1) < upProb ) ? 0 : 1;
	
	// accept-reject according to the efficiency 
	if ( gRandom->Uniform(0,1) > geff[isDn] ) continue; 

	// handle mis-id, according to the diagonal element of the
	// confusion matrix corresponding to the generated class
	double vz = 0; 
	if ( gRandom->Uniform(0,1) < hconf[isDn]) 
	  vz = (isDn==0)? 0.5 : -0.5 ;      
	else 
	  vz = (isDn==0)? -0.5 : 0.5 ;
	hdata.Fill(vz);       
      }

      // add cosmics    
      int muCosm = gRandom->Poisson(cosm);
      double vz ;
      // split in up and down classes 
      for (int i=0; i<muCosm; i++){
	vz = (gRandom->Uniform(0,1) > cdistDn) ? 0.5 : -0.5;
	hdata.Fill(vz);       
      }
      /* ..... end simulation ..... */ 

      /* ..... begin fit ..... */ 

      RooDataHist data("data","data",RooArgList(z),&hdata);  

      RooRealVar nsig("nsig", "number of hbar events", Ngen*Mbal, 0., Ngen);
      RooRealVar nbkg("nbkg", "number of cosm events", cosm);
      RooRealVar nasy("nasy", "up-down asymmetry"   , gRandom->Uniform(-0.99,0.99) , -2., 2.); 
      RooFormulaVar nUp("nUp", "nUp", "0.5*@0*(1.+@1)*@2" , RooArgList(nsig,nasy,effUp) ) ;  
      RooFormulaVar nDn("nDn", "nDn", "0.5*@0*(1.-@1)*@2" , RooArgList(nsig,nasy,effDn) ); 
      RooAddPdf model("model", "model", RooArgList(pbarUp,pbarDn,pcosm), RooArgList(nUp,nDn,nbkg));

      //    model.fitTo(data); // this would work, but let's say we prefer to go interactive 
      
      RooAbsReal* nll = model.createNLL(data, Extended());
      RooMinimizer m(*nll);    
      m.setVerbose(false); 
      int status = m.migrad();    // find minimum 
      m.hesse();   // use second derivatives
      m.minos();   // "graphical" uncertainties - handle non-parabolic NLL and correlations 

      // read the results of the fit that have been back-propagated to the fit parameters
      double vgen = asym; 
      double vfit = nasy.getVal() ; 
      double efit = nasy.getError() ; 
      double hefit = nasy.getErrorHi() ; 
      double lefit = nasy.getErrorLo() ; 
      double pull ;
      bool useAsymErrors= false; 
      if(useAsymErrors){
	if (vfit > vgen )
	  pull = (vgen-vfit)/lefit; 
	else 
	  pull = (vfit-vgen)/hefit;
      }
      else {
	pull = (vfit-vgen)/efit;
      }

      if (status==0)
	hpull.Fill(pull);

      // contour for the curious one
      if(0)     m.contour(nsig,nasy)->Draw();

      // r = m.save();  // this would save a snapshot of the fit results

      /* ..... end fit ..... */ 

    }

    // bookkeeping 
    TFitResultPtr r =  hpull.Fit("gaus","LS");    
    gasym.push_back(asym); 
    par1.push_back(r->Parameter(1));
    err1.push_back(r->ParError(1));
    par2.push_back(r->Parameter(2));
    err2.push_back(r->ParError(2));

  }

  // bookkeeping 
  for ( int i=0; i<gasym.size(); i++ )
    std::cout << gasym.at(i) << ' '
	      << par1.at(i) << ' '
	      << err1.at(i) << ' '
	      << par2.at(i) << ' '
	      << err2.at(i) << endl;
    
}

