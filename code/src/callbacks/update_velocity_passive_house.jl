# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    UpdateVelocityCallback - passive house

    Callback performs explicit time steps for the velocity v and pressure p0.
    If the callback is not used, both quantities will remain constant in time!

    CarpenterKennedy2N54() is used to perform the time integration for v and p0.
    The air velocity u(t,x) (i.e. "v1") is computed using the integrated v.

    Modifications compared to the TermiteMoundInducedAirflowTrixi.jl version:
        - GaussQuadrature weights precomputed
        - Visual nodes are tracked in Callback cache
        - T_inside is computed at each visual node
        - Current solution state at visual nodes will be saved in an output.jl file
"""

mutable struct UpdateVelocityCallback{Vis_count}
    a::Vis_count
end

function Base.show(io::IO, cb::DiscreteCallback{<:Any, <:UpdateVelocityCallback})
    @nospecialize cb # reduce precompilation time

    update_velocity_callback = cb.affect!
    @unpack a, b = update_velocity_callback
    print(io, "UpdateVelocityCallback(a = ", a, ")")
end

function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:Any, <:UpdateVelocityCallback})
    @nospecialize cb # reduce precompilation time

    if get(io, :compact, false)
        show(io, cb)
    else
        update_velocity_callback = cb.affect!

        setup = [
            "Vis Count" => update_velocity_callback.a#,
            #"Error" => update_velocity_callback.b
        ]
        Trixi.summary_box(io, "UpdateVelocityCallback", setup)
    end
end

function UpdateVelocityCallback( ;a=1::Int)
    # Convert plain real numbers to functions for unified treatment
    a_conv = isa(a, Real) ? Returns(a) : a
    update_velocity_callback = UpdateVelocityCallback{typeof(a_conv)}(a_conv)

    DiscreteCallback(condition, update_velocity_callback; save_positions = (false, false))
end

# callback always activated
@inline function condition(u, t, integrator)
    return true
end

# This method is called as callback during the time integration.
@inline function (update_velocity_callback::UpdateVelocityCallback)(integrator)
    """
    "update_velocity_callback" updates the velocity "v1" and pressure "p0" after each time step.
    This function is essential to solve the reformulated asymptotic model.

    Note: If the callback is not used, both "v1" and "p0" will remain constant in time!

    The CarpenterKennedy2N54() method is used to perform the time integration for v.
    Then, the air velocity u(t,x) (i.e. "v1") is computed using the integrated value of v.
    Finally, CarpenterKennedy2N54() is used again for "p0" and the state of the integrator is updated.
    """
    #-----------------------
    #println(fieldnames(typeof(integrator.sol.prob.p.solver.basis)))
    original_nodes = integrator.sol.prob.p.cache.elements.node_coordinates
    nodes = Float64.(vec(original_nodes[1, :, :]))
    L_nodes = length(nodes)
    t = LinRange(nodes[1], nodes[end], L_nodes)
    ti = t[1]:0.01:t[end]
    #-----------------------
    equations = integrator.sol.prob.p.equations
    tspan = integrator.sol.prob.tspan
    #-----------------------
    nodes_unique, idx_unique = unique_idx(nodes, equations)
    L_nodes_unique = length(nodes_unique)
    weights_unique = if L_nodes_unique == 97 # polydeg = 3, initial_refinement_level = 5
                        [0.0003902666109712807, 0.0009080731991050198, 0.001425704621606528, 0.0019419226647446474, 0.002456138131083014, 0.0029678078153941114, 0.003476398048234704, 0.003981379998932748, 0.004482229088349001, 0.0049784252135420225, 0.005469453179597975, 0.005954803192665379, 0.006433971371246692, 0.006906460260926521, 0.0073717793466184746, 0.00782944555957846, 0.008278983777671228, 0.0087199273179039, 0.00915181842048207, 0.009574208723764065, 0.009986659729554022, 0.010388743258213285, 0.010780041893095858, 0.011160149413833564, 0.011528671218012999, 0.011885224730801534, 0.012229439802093922, 0.012560959090765027, 0.012879438435628393, 0.01318454721271467, 0.01347596867849822, 0.01375340029871517, 0.014016554062431337,
                         0.014265156781033591, 0.014498950371834211, 0.014717692125993657, 0.014921154960483511, 0.015109127653828007, 0.01528141506537929, 0.015437838337898824, 0.015578235083234521, 0.01570245955090086, 0.015810382779387008, 0.015901892730035744, 0.01597689440335432, 0.016035309937636397, 0.016077078689792752, 0.01610215729830675, 0.016110519728249353, 0.01610215729830675, 0.016077078689792752, 0.016035309937636397, 0.01597689440335432, 0.015901892730035744, 0.015810382779387008, 0.01570245955090086, 0.015578235083234521, 0.015437838337898824, 0.01528141506537929, 0.015109127653828007, 0.014921154960483511, 0.014717692125993657, 0.014498950371834211, 0.014265156781033591, 0.014016554062431337, 0.01375340029871517,
                         0.01347596867849822, 0.01318454721271467, 0.012879438435628393, 0.012560959090765027, 0.012229439802093922, 0.011885224730801534, 0.011528671218012999, 0.011160149413833564, 0.010780041893095858, 0.010388743258213285, 0.009986659729554022, 0.009574208723764065, 0.00915181842048207, 0.0087199273179039, 0.008278983777671228, 0.00782944555957846, 0.0073717793466184746, 0.006906460260926521, 0.006433971371246692, 0.005954803192665379, 0.005469453179597975, 0.0049784252135420225, 0.004482229088349001, 0.003981379998932748, 0.003476398048234704, 0.0029678078153941114, 0.002456138131083014, 0.0019419226647446474, 0.001425704621606528, 0.0009080731991050198, 0.0003902666109712807]
                    elseif L_nodes_unique == 193 # polydeg = 3, initial_refinement_level = 6
                        [9.909332140972629e-5, 0.00023064504117904569, 0.00036233080939786975, 0.0004939379961782751, 0.0006254183845390792, 0.0007567349372492142, 0.0008878524094827533, 0.0010187360254398647, 0.0011493511992073614, 0.0012796634627200594, 0.0014096384474302074, 0.001539241882767983, 0.0016684396006277759, 0.001797197542254745, 0.0019254817661646755, 0.0020532584565321195, 0.002180493931794507, 0.002307154653351663, 0.0024332072342993765, 0.002558618448163875, 0.002683355237618114, 0.0028073847231681066, 0.0029306742118014985, 0.003053191205592817, 0.003174903410261095, 0.0032957787436763385, 0.0034157853443117963, 0.003534891579639248, 0.0036530660544647586, 0.0037702776192024613, 0.003886495378084007, 0.004001688697301435,
                         0.00411582721308119, 0.004228880839687151, 0.004340819777350477, 0.004451614520124193, 0.004561235863660366, 0.004669654912907864, 0.00477684308972862, 0.0048827721404304, 0.004987414143214059, 0.005090741515533367, 0.005192727021365385, 0.0052933437783895575, 0.005392565265073551, 0.0054903653276640145, 0.005586718187080403, 0.00568159844571005, 0.0057749810941026825, 0.005866841517562637, 0.0059571555026370175, 0.006045899243498097, 0.006133049348218276, 0.006218582844935926, 0.006302477187910551, 0.006384710263465575, 0.006465260395817285, 0.006544106352788334, 0.006621227351404317, 0.006696603063371939, 0.006770213620437348, 0.006842039619623177, 0.006912062128342985, 0.0069802626893916775, 0.007046623325810629,
                         0.0071111265456262245, 0.0071737553464605655, 0.0072344932200131065, 0.007293324156412073, 0.007350232648434506, 0.0074052036955937935, 0.007458222808093639, 0.0075092760106474305, 0.007558349846161979, 0.007605431379284655, 0.007650508199813031, 0.007693568425966059, 0.007734600707516006, 0.0077735942287802395, 0.007810538711472141, 0.007845424417410343, 0.007878242151085619, 0.007908983262084722, 0.00793763964737055, 0.007964203753418, 0.007988668578205037, 0.00801102767305831, 0.008031275144352989, 0.008049405655066247, 0.008065414426184041, 0.008079297237960827, 0.008091050431031824, 0.008100670907377587, 0.008108156131140623, 0.008113504129293809, 0.008116713492160452, 0.008117783373785893, 0.008116713492160452,
                         0.008113504129293809, 0.008108156131140623, 0.008100670907377587, 0.008091050431031824, 0.008079297237960827, 0.008065414426184041, 0.008049405655066247, 0.008031275144352989, 0.00801102767305831, 0.007988668578205037, 0.007964203753418, 0.00793763964737055, 0.007908983262084722, 0.007878242151085619, 0.007845424417410343, 0.007810538711472141, 0.0077735942287802395, 0.007734600707516006, 0.007693568425966059, 0.007650508199813031, 0.007605431379284655, 0.007558349846161979, 0.0075092760106474305, 0.007458222808093639, 0.0074052036955937935, 0.007350232648434506, 0.007293324156412073, 0.0072344932200131065, 0.0071737553464605655, 0.0071111265456262245, 0.007046623325810629, 0.0069802626893916775, 0.006912062128342985,
                         0.006842039619623177, 0.006770213620437348, 0.006696603063371939, 0.006621227351404317, 0.006544106352788334, 0.006465260395817285, 0.006384710263465575, 0.006302477187910551, 0.006218582844935926, 0.006133049348218276, 0.006045899243498097, 0.0059571555026370175, 0.005866841517562637, 0.0057749810941026825, 0.00568159844571005, 0.005586718187080403, 0.0054903653276640145, 0.005392565265073551, 0.0052933437783895575, 0.005192727021365385, 0.005090741515533367, 0.004987414143214059, 0.0048827721404304, 0.00477684308972862, 0.004669654912907864, 0.004561235863660366, 0.004451614520124193, 0.004340819777350477, 0.004228880839687151, 0.00411582721308119, 0.004001688697301435, 0.003886495378084007, 0.0037702776192024613,
                         0.0036530660544647586, 0.003534891579639248, 0.0034157853443117963, 0.0032957787436763385, 0.003174903410261095, 0.003053191205592817, 0.0029306742118014985, 0.0028073847231681066, 0.002683355237618114, 0.002558618448163875, 0.0024332072342993765, 0.002307154653351663, 0.002180493931794507, 0.0020532584565321195, 0.0019254817661646755, 0.001797197542254745, 0.0016684396006277759, 0.001539241882767983, 0.0014096384474302074, 0.0012796634627200594, 0.0011493511992073614, 0.0010187360254398647, 0.0008878524094827533, 0.0007567349372492142, 0.0006254183845390792, 0.0004939379961782751, 0.00036233080939786975, 0.00023064504117904569, 9.909332140972629e-5]
                    elseif L_nodes_unique == 129 # polydeg = 4, initial_refinement_level = 5
                        [0.0002212339709146965, 0.0005148642230981119, 0.0008086265278392765, 0.001101950759048347, 0.0013946340938898777, 0.0016864989753123123, 0.0019773722341056787, 0.0022670822149262713, 0.0025554582334623138, 0.0028423304956234526, 0.00312753013622307, 0.003410889294675956, 0.0036922412036227, 0.003971420282333401, 0.004248262231786163, 0.004522604130106866, 0.0047942845277552095, 0.0050631435421366775, 0.005329022951452759, 0.0055917662876652525, 0.005851218928482389, 0.00610722818829149, 0.0063596434079723095, 0.006608316043530862, 0.0068530997534969875, 0.007093850485031449, 0.007330426558690031, 0.007562688751793514, 0.00779050038035376, 0.008013727379507106, 0.008232238382407333, 0.008445904797531604, 0.008654600884353622,
                         0.008858203827339407, 0.00905659380822199, 0.009249654076512492, 0.009437271018205975, 0.009619334222641643, 0.00979573654747801, 0.009966374181744772, 0.01013114670693422, 0.010289957156096333, 0.010442712070902657, 0.010589321556645426, 0.0107296993351396, 0.010863762795496555, 0.010991433042739694, 0.011112634944233262, 0.011227297173897092, 0.011335352254181188, 0.011436736595775583, 0.01153139053503194, 0.011619258369074946, 0.011700288388582916, 0.011774432908218129, 0.011841648294689169, 0.011901894992428655, 0.011955137546871265, 0.012001344625318378, 0.012040489035377043, 0.01207254774096242, 0.01209750187585425, 0.012115336754799465, 0.01212604188215428, 0.012129610958060769, 0.01212604188215428, 0.012115336754799465,
                         0.01209750187585425, 0.01207254774096242, 0.012040489035377043, 0.012001344625318378, 0.011955137546871265, 0.011901894992428655, 0.011841648294689169, 0.011774432908218129, 0.011700288388582916, 0.011619258369074946, 0.01153139053503194, 0.011436736595775583, 0.011335352254181188, 0.011227297173897092, 0.011112634944233262, 0.010991433042739694, 0.010863762795496555, 0.0107296993351396, 0.010589321556645426, 0.010442712070902657, 0.010289957156096333, 0.01013114670693422, 0.009966374181744772, 0.00979573654747801, 0.009619334222641643, 0.009437271018205975, 0.009249654076512492, 0.00905659380822199, 0.008858203827339407, 0.008654600884353622, 0.008445904797531604, 0.008232238382407333, 0.008013727379507106, 0.00779050038035376,
                         0.007562688751793514, 0.007330426558690031, 0.007093850485031449, 0.0068530997534969875, 0.006608316043530862, 0.0063596434079723095, 0.00610722818829149, 0.005851218928482389, 0.0055917662876652525, 0.005329022951452759, 0.0050631435421366775, 0.0047942845277552095, 0.004522604130106866, 0.004248262231786163, 0.003971420282333401, 0.0036922412036227, 0.003410889294675956, 0.00312753013622307, 0.0028423304956234526, 0.0025554582334623138, 0.0022670822149262713, 0.0019773722341056787, 0.0016864989753123123, 0.0013946340938898777, 0.001101950759048347, 0.0008086265278392765, 0.0005148642230981119, 0.0002212339709146965]
                    elseif L_nodes_unique == 257 # polydeg = 4, initial_refinement_level = 6
                        [5.595735072800878e-5, 0.0001302499779008848, 0.0002046332414176567, 0.00027899560273440317, 0.00035331835525796154, 0.00042758909223348287, 0.0005017964023398473, 0.0005759291188913339, 0.0006499761587117614, 0.0007239264779627559, 0.0007977690583087565, 0.0008714929025734152, 0.0009450870338095053, 0.0010185404957361811, 0.0010918423537727624, 0.0011649816963510914, 0.0012379476363650744, 0.0013107293126904055, 0.0013833158917409137, 0.0014556965690438923, 0.0015278605708246855, 0.0015997971555949716, 0.001671495615741381, 0.0017429452791123564, 0.0018141355106018888, 0.0018850557137291442, 0.0019556953322133327, 0.0020260438515432425, 0.0020960908005410125, 0.0021658257529198143, 0.002235238328835054, 0.0023043181964288533,
                         0.0023730550733675093, 0.002441438728371671, 0.002509458982738994, 0.002577105711859019, 0.0026443688467200355, 0.0027112383754077066, 0.0027777043445952145, 0.0028437568610247066, 0.0029093860929798173, 0.0029745822717490334, 0.0030393356930796967, 0.0031036367186224227, 0.003167475777365708, 0.003230843367060522, 0.003293730055634667, 0.0033561264825967036, 0.0034180233604292113, 0.003479411475971197, 0.003540281691789435, 0.0036006249475385448, 0.003660432261309578, 0.0037196947309669484, 0.0037784035354734833, 0.0038365499362033977, 0.003894125278243013, 0.003951120991679013, 0.004007528592874039, 0.004063339685729454, 0.0041185459629350845, 0.004173139207205714, 0.004227111292504219, 0.004280454185251097, 0.004333159945520233,
                         0.004385220728220743, 0.004436628784264679, 0.004487376461720467, 0.004537456206951863, 0.004586860565742297, 0.004635582184404404, 0.004683613810874594, 0.00473094829579251, 0.004777578593565182, 0.0048234977634157294, 0.0048686989704165, 0.004913175486506408, 0.004956920691492387, 0.0049999280740347886, 0.005042191232616583, 0.005083703876496184, 0.00512445982664383, 0.0051644530166612985, 0.005203677493684877, 0.005242127419271427, 0.005279797070267411, 0.005316680839660778, 0.005352773237415545, 0.005388068891288974, 0.0054225625476312074, 0.005456249072167261, 0.005489123450761246, 0.005521180790162714, 0.005552416318734987, 0.0055828253871654165, 0.005612403469157405, 0.005641146162104146, 0.005669049187743923,
                         0.005696108392796935, 0.005722319749583475, 0.005747679356623465, 0.005772183439217155, 0.005795828350006986, 0.005818610569520493, 0.00584052670669416, 0.005861573499378171, 0.005881747814821975, 0.0059010466501405725, 0.005919467132761511, 0.005937006520852433, 0.005953662203729207, 0.005969431702244507, 0.005984312669156835, 0.0059983028894798965, 0.0060114002808122945, 0.006023602893647497, 0.006034908911663998, 0.006045316651995682, 0.006054824565482319, 0.006063431236900139, 0.006071135385172499, 0.006077935863560544, 0.006083831659833922, 0.00608882189642144, 0.006092905830541683, 0.00609608285431358, 0.006098352494846883, 0.00609971441431256, 0.006100168409993072, 0.00609971441431256, 0.006098352494846883, 0.00609608285431358,
                         0.006092905830541683, 0.00608882189642144, 0.006083831659833922, 0.006077935863560544, 0.006071135385172499, 0.006063431236900139, 0.006054824565482319, 0.006045316651995682, 0.006034908911663998, 0.006023602893647497, 0.0060114002808122945, 0.0059983028894798965, 0.005984312669156835, 0.005969431702244507, 0.005953662203729207, 0.005937006520852433, 0.005919467132761511, 0.0059010466501405725, 0.005881747814821975, 0.005861573499378171, 0.00584052670669416, 0.005818610569520493, 0.005795828350006986, 0.005772183439217155, 0.005747679356623465, 0.005722319749583475, 0.005696108392796935, 0.005669049187743923, 0.005641146162104146, 0.005612403469157405, 0.0055828253871654165, 0.005552416318734987, 0.005521180790162714, 
                         0.005489123450761246, 0.005456249072167261, 0.0054225625476312074, 0.005388068891288974, 0.005352773237415545, 0.005316680839660778, 0.005279797070267411, 0.005242127419271427, 0.005203677493684877, 0.0051644530166612985, 0.00512445982664383, 0.005083703876496184, 0.005042191232616583, 0.0049999280740347886, 0.004956920691492387, 0.004913175486506408, 0.0048686989704165, 0.0048234977634157294, 0.004777578593565182, 0.00473094829579251, 0.004683613810874594, 0.004635582184404404, 0.004586860565742297, 0.004537456206951863, 0.004487376461720467, 0.004436628784264679, 0.004385220728220743, 0.004333159945520233, 0.004280454185251097, 0.004227111292504219, 0.004173139207205714, 0.0041185459629350845, 0.004063339685729454,
                         0.004007528592874039, 0.003951120991679013, 0.003894125278243013, 0.0038365499362033977, 0.0037784035354734833, 0.0037196947309669484, 0.003660432261309578, 0.0036006249475385448, 0.003540281691789435, 0.003479411475971197, 0.0034180233604292113, 0.0033561264825967036, 0.003293730055634667, 0.003230843367060522, 0.003167475777365708, 0.0031036367186224227, 0.0030393356930796967, 0.0029745822717490334, 0.0029093860929798173, 0.0028437568610247066, 0.0027777043445952145, 0.0027112383754077066, 0.0026443688467200355, 0.002577105711859019, 0.002509458982738994, 0.002441438728371671, 0.0023730550733675093, 0.0023043181964288533, 0.002235238328835054, 0.0021658257529198143, 0.0020960908005410125, 0.0020260438515432425,
                         0.0019556953322133327, 0.0018850557137291442, 0.0018141355106018888, 0.0017429452791123564, 0.001671495615741381, 0.0015997971555949716, 0.0015278605708246855, 0.0014556965690438923, 0.0013833158917409137, 0.0013107293126904055, 0.0012379476363650744, 0.0011649816963510914, 0.0010918423537727624, 0.0010185404957361811, 0.0009450870338095053, 0.0008714929025734152, 0.0007977690583087565, 0.0007239264779627559, 0.0006499761587117614, 0.0005759291188913339, 0.0005017964023398473, 0.00042758909223348287, 0.00035331835525796154, 0.00027899560273440317, 0.0002046332414176567, 0.0001302499779008848, 5.595735072800878e-5]
                    else
                        x, w = gausslegendre(L_nodes_unique)
                        weights_unique = w./2
                    end
    weights = zeros(L_nodes)
    weights[idx_unique] = weights_unique
    #-----------------------
    A_nodes = [A(x, equations) for x in nodes]
    A_x_nodes = [A_x(x, equations) for x in nodes]
    A_nodes_inv = [inv(A(x, equations)) for x in nodes]
    A_sqrt_A_inv = [inv(sqrt(A(x, equations))) for x in nodes]
    I_w_nodes = [I_w(x, equations) for x in nodes]
    #-----------------------
    t_prev = integrator.tprev
    rho_prev = integrator.uprev[1:5:end-4]
    v1_prev = integrator.uprev[2:5:end-3]
    v1_prev_LI = bspline2linear(nodes, v1_prev, t, ti, equations)
    v1_dx_prev = [Interpolations.gradient(v1_prev_LI, nodes[i])[1] for i in 2:L_nodes-1]
    v1_dx_prev = vcat(Interpolations.gradient(v1_prev_LI, 0.0001)[1], vcat(v1_dx_prev, Interpolations.gradient(v1_prev_LI, 0.9999)[1]))
    v_prev = v1_prev[1] * A(0, equations)
    Ti_prev = integrator.uprev[4:5:end-1]
    p0_prev = integrator.uprev[3:5:end-2]
    T_prev = p0_prev ./ rho_prev
    Tu_prev = [T_u(t_prev, x, y, equations) * equations.t_ref for (x, y) in zip(nodes, Ti_prev)] 
    #-----------------------
    # \frac{\partial p_0}{\partial t} = - \gamma p_0 \frac{\partial u}{\partial x} - \gamma \frac{\textrm{A}_x}{\textrm{A}} u p_0 - \frac{k_w}{\textrm{A} \sqrt{\textrm{A}}} \left(T - \textrm{T}_\textrm{u} \right)
    #p0_dt_prev = - equations.γ * p0_prev[1] * v1_dx_prev[1] - equations.γ * A_x(0.0,equations) / A(0.0,equations) * v1_prev[1] * p0_prev[1] - I_w(0.0, equations) * equations.k_w / (A(0.0,equations) * sqrt(A(0.0,equations))) * (T_prev[1] - Tu_prev[1])
    p0_dt_prev = .- equations.γ .* p0_prev .* v1_dx_prev .- equations.γ .* A_x_nodes ./ A_nodes .* v1_prev .* p0_prev .- I_w_nodes .* equations.k_w .* A_sqrt_A_inv .* (T_prev .- Tu_prev)
    p0_dt_prev = sum((p0_dt_prev) .* weights)
    #-----------------------
    #\frac{\partial v}{\partial t} = \frac{1}{\int_0^1 \frac{\rho}{A} \, dy} \left[ \int_0^1 -\rho u u_x  - \rho u \left( \beta \eta - \beta\left(1-\eta\right) \vert u \vert \right) - \frac{\textrm{h}_x}{Fr^2}(\rho - \rho_{h_0}) \, dy \right]
    I_inv = inv.(sum((rho_prev.*A_nodes_inv) .* weights))
    beta = [equations.β ./ A(x,equations) for x in nodes]
    f = - rho_prev .* v1_prev .* v1_dx_prev - beta .* rho_prev .* v1_prev .* ( equations.η .- (1 - equations.η) .* abs.(v1_prev)) - h_x.(nodes, equations) .* (rho_prev .- equations.ρₕ₀) ./ equations.Fr²
    F = sum((f) .* weights)
    v_dt_prev = I_inv .* F
    #-----------------------
    t_now = integrator.t
    #-----------------------
    rho = integrator.u[1:5:end-4]
    if any(isnan, rho)
        #println(rho)
        error("NaN detected in rho!")
    end
    if t_prev == t_now
        error("Time step → 0!")
    end
    Ti = integrator.u[4:5:end-1]
    p0 = integrator.u[3:5:end-2]
    T = p0 ./ rho
    Tu = [T_u(t_now, x, y, equations) * equations.t_ref for (x, y) in zip(nodes, Ti)]
    #-----------------------
    ### Explixit velocity time step via CarpenterKennedy2N54() ###
    direction_prev = sign(v_prev)
    v_exp = v_prev + integrator.dt * v_dt_prev
    direction_exp = sign(v_exp)
    v_dt =  if direction_prev == direction_exp
            v1_exp = v1_prev .+ (v_exp - v_prev) / A(0.0, equations)
            v1_dx_exp = v1_dx_prev
            I_inv = inv.(sum((rho.*A_nodes_inv) .* weights))
            f = - rho .* v1_exp .* v1_dx_exp - beta .* rho .* v1_exp .* ( equations.η .- (1 - equations.η) .* abs.(v1_exp)) - h_x.(nodes, equations) .* (rho .- equations.ρₕ₀) ./ equations.Fr²
            F = sum((f) .* weights)
            I_inv .* F
            else
            v1_exp = reverse( - v1_prev .+ (v_exp + v_prev) / A(0.0, equations))
            v1_exp_LI = bspline2linear(nodes, v1_exp, t, ti, equations)
            v1_dx_exp = [Interpolations.gradient(v1_exp_LI, nodes[i])[1] for i in 1:L_nodes]
            I_inv = inv.(sum((rho.*A_nodes_inv) .* weights))
            f = - rho .* v1_exp .* v1_dx_exp - beta .* rho .* v1_exp .* ( equations.η .- (1 - equations.η) .* abs.(v1_exp)) - h_x.(nodes, equations).* (rho .- equations.ρₕ₀) ./ equations.Fr²
            F = sum((f) .* weights)
            I_inv .* F
    end
    LI = LinearInterpolation([t_prev, t_now], [v_dt_prev, v_dt], extrapolation_bc=Line()) 
    prob = ODEProblem((u, p, t) -> LI(t), v_prev, (t_prev, t_now))
    sol = solve(prob, CarpenterKennedy2N54(williamson_condition = false), dt = integrator.dt)
    v = sol.u[end]
    #-----------------------
    ### update u(t,x) ###
    Q = (- I_w_nodes .* equations.k_w .* A_sqrt_A_inv .* (T_prev .- Tu_prev)) ./ (equations.γ .* p0_prev[1])
    c = sum((Q.*A_nodes) .* weights)
    I_0 = cumsum((Q.*A_nodes.-c) .* weights)
    v1 = v .* A_nodes_inv .+ A_nodes_inv .* I_0
    v1_LI = LinearInterpolation2(nodes, v1, equations)
    v1_dx = [Interpolations.gradient(v1_LI, nodes[i])[1] for i in 1:L_nodes]
    #------------------------
    #p0_dt = - equations.γ * p0[1] * v1_dx[1] - equations.γ * A_x(0.0,equations) / A(0.0,equations) * v1[1] * p0[1] - I_w(0.0, equations) * equations.k_w / (A(0.0,equations) * sqrt(A(0.0,equations))) * (T[1] - Tu[1]) 
    p0_dt = .- equations.γ .* p0 .* v1_dx .- equations.γ .* A_x_nodes ./ A_nodes .* v1 .* p0 .- I_w_nodes .* equations.k_w .* A_sqrt_A_inv .* (T .- Tu)
    p0_dt = sum((p0_dt) .* weights)
    LI = LinearInterpolation([t_prev, t_now], [p0_dt_prev, p0_dt], extrapolation_bc=Line()) 
    prob = ODEProblem((u, p, t) -> LI(t), p0_prev[1], (t_prev, t_now))
    sol = solve(prob, CarpenterKennedy2N54(williamson_condition = false), dt = integrator.dt)
    p0 = sol.u[end]
    #-----------------------
    integrator.u[2:5:end-3] = v1
    integrator.u[3:5:end-2] .= p0
    #-----------------------
    a_now = update_velocity_callback.a()
    a_now = if t_prev == tspan[1]
        1
    else
        a_now
    end
    a_new = write_plot_data(a_now, integrator, tspan, nodes, t, ti, 151, rho, v1, v1_prev, T, Ti, Tu, p0, equations)
    update_velocity_callback.a = isa(a_new, Real) ? Returns(a_new) : a_new
    #-----------------------
    return integrator
end

@inline function write_plot_data(a, integrator, tspan, nodes, t, ti, max_visnodes, rho, v1, v1_prev, T, Ti, Tu, p0, equations::PassiveHouseEquations1D)
    #-----------------------
    vis_index = range(0.0, tspan[end], max_visnodes)
    #-----------------------
    # save output:
    output_dir = joinpath(@__DIR__, "..//..//out//passive_house")
    vis_status = if integrator.tprev ≤ 0.0 && integrator.t > 0.0
        x_s = space2unscaled.(range(nodes[1], nodes[end], 59), equations)
        rho_LI = bspline2linear(nodes, rho, t, ti, equations)
        ρᵣ = 1.17
        rho_s = ρᵣ .* [rho_LI(x) for x in range(nodes[1], nodes[end], 59)]
        v1_LI = bspline2linear(nodes, v1, t, ti, equations)
        v_s = vel2unscaled.([v1_LI(x) for x in range(nodes[1], nodes[end], 59)], equations)
        T_LI = bspline2linear(nodes, T, t, ti, equations)
        T_s = temp2unscaled.([T_LI(x) for x in range(nodes[1], nodes[end], 59)], equations)
        Ti_LI = bspline2linear(nodes, Ti, t, ti, equations)
        Ti_s = temp2unscaled.([Ti_LI(x) for x in range(nodes[1], nodes[end], 59)], equations)
        Tu_LI = bspline2linear(nodes, Tu, t, ti, equations)
        Tu_s = temp2unscaled.([Tu_LI(x) for x in range(nodes[1], nodes[end], 59)], equations)
        Tinside = [temp2unscaled( quadgk(x -> Ti_LI(x), equations.xa, equations.xc)[1] / (equations.xc - equations.xa), equations)]
        string1 = createString1(x_s, time2unscaled(integrator.t, equations), rho_s, v_s, T_s, Ti_s, Tu_s, p0, Tinside)
        open(joinpath(output_dir, "output.jl"), "w") do file
                        write(file, string1)
                        flush(file)
        end
        1.0
    elseif integrator.t > vis_index[a]
        rho_LI = bspline2linear(nodes, rho, t, ti, equations)
        ρᵣ = 1.17
        rho_s = ρᵣ .* [rho_LI(x) for x in range(nodes[1], nodes[end], 59)]
        v1_LI = bspline2linear(nodes, v1, t, ti, equations)
        v_s = vel2unscaled.([v1_LI(x) for x in range(nodes[1], nodes[end], 59)], equations)
        T_LI = bspline2linear(nodes, T, t, ti, equations)
        T_s = temp2unscaled.([T_LI(x) for x in range(nodes[1], nodes[end], 59)], equations)
        Ti_LI = bspline2linear(nodes, Ti, t, ti, equations)
        Ti_s = temp2unscaled.([Ti_LI(x) for x in range(nodes[1], nodes[end], 59)], equations)
        Tu_LI = bspline2linear(nodes, Tu, t, ti, equations)
        Tu_s = temp2unscaled.([Tu_LI(x) for x in range(nodes[1], nodes[end], 59)], equations)
        Tinside = [temp2unscaled( quadgk(x -> Ti_LI(x), equations.xa, equations.xc)[1] / (equations.xc - equations.xa), equations)]
        string2 = createString2(time2unscaled(integrator.t, equations), rho_s, v_s, T_s, Ti_s, Tu_s, p0, Tinside) 
        open(joinpath(output_dir, "output.jl"), "a") do file
                        write(file, string2)
                        flush(file)
        end
        1.0
    elseif integrator.t == tspan[end]
        rho_LI = bspline2linear(nodes, rho, t, ti, equations)
        ρᵣ = 1.17
        rho_s = ρᵣ .* [rho_LI(x) for x in range(nodes[1], nodes[end], 59)]
        v1_LI = bspline2linear(nodes, v1, t, ti, equations)
        v_s = vel2unscaled.([v1_LI(x) for x in range(nodes[1], nodes[end], 59)], equations)
        T_LI = bspline2linear(nodes, T, t, ti, equations)
        T_s = temp2unscaled.([T_LI(x) for x in range(nodes[1], nodes[end], 59)], equations)
        Ti_LI = bspline2linear(nodes, Ti, t, ti, equations)
        Ti_s = temp2unscaled.([Ti_LI(x) for x in range(nodes[1], nodes[end], 59)], equations)
        Tu_LI = bspline2linear(nodes, Tu, t, ti, equations)
        Tu_s = temp2unscaled.([Tu_LI(x) for x in range(nodes[1], nodes[end], 59)], equations)
        Tinside = [temp2unscaled( quadgk(x -> Ti_LI(x), equations.xa, equations.xc)[1] / (equations.xc - equations.xa), equations)]
        string2 = createString2(time2unscaled(integrator.t, equations), rho_s, v_s, T_s, Ti_s, Tu_s, p0, Tinside) 
        string3 = createString3()
        open(joinpath(output_dir, "output.jl"), "a") do file
                        write(file, string2 * string3)
                        flush(file)
        end
        1.0
    else
       0.0
    end
    #-----------------------
    if vis_status == 1.0
        a_new = a + 1
    else
        a_new = a
    end
    #-----------------------
    return a_new
end

@inline function writeString(x,z)
    y = round.(x, digits=z) 
    y_str = join(y, ", ")  
    return "[" * y_str * "]" 
end

@inline function createString1(nodes, time, rho_s, v_s, T_s, Ti_s, Tu_s, p0, Tinside)
    xnodes_str = writeString(nodes, 8)
    t_str = writeString(time, 8)
    rho_str = writeString(rho_s, 5)
    v_str = writeString(v_s, 5)
    T_str = writeString(T_s,5)
    Ti_str = writeString(Ti_s, 5)
    Tu_str = writeString(Tu_s, 5)
    p0_str = writeString(p0, 5)
    Tinside_str = writeString(Tinside, 5)
    string =  "## output.jl ##\n#\n# Passive_house_simulation\n#\n#------------------------------------------------------------------------------\n### read output ###\n@muladd @inline function read_output()\n    nodes = " * xnodes_str *"\n    #\n    t = " * t_str * "\n    rho = " * rho_str * "\n    v = " * v_str * "\n    T = " * T_str * "\n    Ti = " * Ti_str * "\n    Tu = " * Tu_str * "\n    p0 = " * p0_str * "\n    Tinside = " * Tinside_str * "\n    #\n"
    return string
end

@inline function createString2(time, rho_s, v_s, T_s, Ti_s, Tu_s, p0, Tinside) 
    t_str = writeString(time, 8)
    rho_str = writeString(rho_s, 5)
    v_str = writeString(v_s, 5)
    T_str = writeString(T_s,5)
    Ti_str = writeString(Ti_s, 5)
    Tu_str = writeString(Tu_s, 5)
    p0_str = writeString(p0, 5)
    Tinside_str = writeString(Tinside, 5)
    string = "    t =  vcat(t," * t_str *")\n    rho = vcat(rho," * rho_str * ")\n    v = vcat(v," * v_str * ")\n    T =  vcat(T," * T_str * ")\n    Ti =  vcat(Ti," * Ti_str * ")\n    Tu =  vcat(Tu," * Tu_str * ")\n    p0 =  vcat(p0," * p0_str * ")\n    Tinside =  vcat(Tinside," * Tinside_str * ")\n      #\n"
    return string
end   

@inline function createString3()
    string = "    return nodes, t, rho, v, T, Ti, Tu, p0, Tinside\nend\n#------------------------------------------------------------------------------\n"
    return string
end  

end # @muladd
